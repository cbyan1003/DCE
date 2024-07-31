import torch
from torch import nn as nn
from models.block import UnaryBlock
from utils.transformations import transform_points_Rt
from .alignment import align
from .backbones import *
from .correspondence import get_correspondences_1, get_correspondences_2
from .model_util import get_grid, grid_to_pointcloud, points_to_ndc
from .renderer import PointsRenderer
from monai.networks.nets import UNet
from pytorch3d.ops.knn import knn_points
from models.correspondence import calculate_ratio_test, get_topk_matches
import warnings
import torch.nn.functional as F
from .sd_model.sd_img import control_extractor
from .sd_model.capture import capture
from .hyper_emb import Hyper_Emb_Decoder, SelfAttention
import gc
import time
warnings.filterwarnings("ignore")

def project_rgb(pc_0in1_X, rgb_src, renderer):
    # create rgb_features
    B, _, H, W = rgb_src.shape
    rgb_src = rgb_src.view(B, 3, H * W)
    rgb_src = rgb_src.permute(0, 2, 1).contiguous()

    # Rasterize and Blend
    project_0in1 = renderer(pc_0in1_X, rgb_src)

    return project_0in1["feats"]


#baseline
class PCReg(nn.Module):
    def __init__(self, cfg):
        super(PCReg, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False
        self.encode = ResNetEncoder(chan_in, feat_dim, pretrained)
        self.decode = ResNetDecoder(feat_dim, 3, nn.Tanh(), pretrained)

        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        self.align_cfg = cfg

    def forward(self, rgbs, K, deps, vps=None):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        if vps is not None:
            # Drop first viewpoint -- assumed to be identity transformation
            vps = vps[1:]
        elif self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                corr_i = get_correspondences(
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                )
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1 : n_views], dim=1)
                pcs_F_joint = torch.cat(pcs_F[0:i] + pcs_F[i + 1 : n_views], dim=1)
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1 : n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint, K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint))

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_decode_{i}"] = self.decode(proj_F_i)
            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless

        return output

    def forward_pcreg(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)
        output = {}

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert feats[0].shape[-1] == deps[0].shape[-1], "Same size"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]

        vps = []
        cor_loss = []
        for i in range(1, n_views):
            corr_i = get_correspondences(
                P1=pcs_F[0],
                P2=pcs_F[i],
                P1_X=pcs_X[0],
                P2_X=pcs_X[i],
                num_corres=self.num_corres,
                ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
            )
            Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

            vps.append(Rt_i)
            cor_loss.append(cor_loss_i)

            # add for visualization
            output[f"corres_0{i}"] = corr_i
            output[f"vp_{i}"] = Rt_i

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)

        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        return output

    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True,)
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
            feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None


#replace recurrent resblock in baseline with 2 resnet blocks
from .backbones import ResNetEncoder_modified



class PCReg_KPURes18_MSF(nn.Module):
    def __init__(self, cfg):
        super(PCReg_KPURes18_MSF, self).__init__()
        # set encoder decoder
        chan_in = 3
        self.cfg = cfg
        feat_dim = cfg.feat_dim

        # No imagenet pretraining
        pretrained = False

        encode_I = URes18Encoder1(chan_in, feat_dim, pretrained)
        encode_P = KPFCN(cfg)
        self.cnn_pre_stages = nn.Sequential(
            encode_I.inconv,
            encode_I.layer1
        )
        self.pcd_pre_stages = nn.ModuleList()
        for i in range(0,2):
            self.pcd_pre_stages.append(encode_P.encoder_blocks[i])

        self.cnn_ds_0 = encode_I.layer2
        self.pcd_ds_0 = nn.ModuleList()
        for i in range(2,5):
            self.pcd_ds_0.append(encode_P.encoder_blocks[i])

        # self.cnn_ds_1 = encode_I.layer3
        # self.pcd_ds_1 = nn.ModuleList()
        # for i in range(5,8):
        #     self.pcd_ds_1.append(encode_P.encoder_blocks[i])

        # self.cnn_up_0 = encode_I.up1
        # self.pcd_up_0 = nn.ModuleList()
        # for i in range(0, 2):
        #     self.pcd_up_0.append(encode_P.decoder_blocks[i])

        self.cnn_up_1 = nn.Sequential(
            encode_I.up2,
            encode_I.outconv
        )
        self.pcd_up_1 = nn.ModuleList()
        for i in range(2, 4):
            self.pcd_up_1.append(encode_P.decoder_blocks[i])

        # 0
        self.p2i_downsample_0 = nn.ModuleList()
        self.p2i_downsample_0.append(
            UnaryBlock(128, 64, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.p2i_downsample_0.append(
            nn.Sequential(
                nn.Conv2d(64*2, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        )
        
        self.p2i_downsample_1 = nn.ModuleList()
        self.p2i_downsample_1.append(
            UnaryBlock(128*2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.p2i_downsample_1.append(
            nn.Sequential(
                nn.Conv2d(128*2, 128, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        
        self.i2p_downsample_1 = nn.ModuleList()
        self.i2p_downsample_1.append(
            UnaryBlock(128, 128*2, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.i2p_downsample_1.append(
            UnaryBlock(256 * 2, 256, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        
        self.i2p_downsample_0 = nn.ModuleList()
        self.i2p_downsample_0.append(
            UnaryBlock(64, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )
        self.i2p_downsample_0.append(
            UnaryBlock(128 * 2, 128, cfg.use_batch_norm, cfg.batch_norm_momentum),
        )

        self.align_cfg = cfg
        self.renderer = PointsRenderer(cfg)
        self.num_corres = cfg.num_correspodances
        self.pointcloud_source = cfg.pointcloud_source
        # self.map = Fusion_CATL(feat_dim)
        self.map = Hyper_Emb_Decoder(
            input_dim = 32,
            output_dim = 64)
        
        self.use_original = cfg.use_original
        if not self.use_original:
            # self.device = torch.device("cuda")# 
            # self.prompt = 'a photo of a room and furniture'
            # # self.capturer = capture(seed = cfg.seed, basic = cfg.basic, 
            # #                             yaml = cfg.yaml, sd_ckpt = cfg.sd_ckpt,
            # #                             cn_ckpt = cfg.cn_ckpt, t = cfg.t)
            # # self.img_processing = img_processor(self.capturer, self.cfg)
            # # self.capturor = capture(load_model = False)
            # self.check_layers = [0,4,6,11]
            # self.pca_dim = 16
            # self.control = control_extractor(load_model=True,
            #             cfg=self.cfg,
            #             seed=cfg.seed, 
            #             t = cfg.t, 
            #             basic= cfg.basic,
            #             yaml=cfg.yaml,
            #             sd_ckpt=cfg.sd_ckpt,
            #             cn_ckpt=cfg.cn_ckpt,
            #             prompt=self.prompt) 
            self.attn = SelfAttention(in_channels = 64)


    def fusion_attn(self, rgb_items, dep_items):
        feat_i = []
        for i in range(len(rgb_items)):
            rgb_feat = rgb_items[i]
            dep_feat = dep_items[i]
            for j in range(len(rgb_feat)):
                B, C, *size = rgb_feat[j].shape
                rgb_feat[j] = F.interpolate(rgb_feat[j], size=(32, 32), mode='bilinear', align_corners=False)
                dep_feat[j] = F.interpolate(dep_feat[j], size=(32, 32), mode='bilinear', align_corners=False)
            feat_i.append(self.attn(torch.cat(list(rgb_feat.values()), dim = 1), torch.cat(list(dep_feat.values()), dim = 1)))
        return feat_i[0], feat_i[1]

    def feature_pca_and_aggregate(self, rgb_items, dep_items):
        feat_i = []
        for i in range(len(rgb_items)):
            rgb_feat = rgb_items[i]
            dep_feat = dep_items[i]
            rgb_feat, dep_feat = self.capturor.merge_feat(rgb_feat, dep_feat, checklayers = self.check_layers, pca_dim = self.pca_dim)
            feat_i.append(rgb_feat + dep_feat)
        return feat_i[0], feat_i[1]
    
    def img_sd_feat(self, img):
        feats = self.control.rgb_feature(img)
        target_w = target_h = self.cfg.img_dim
        feats = {i:feats[i] for i in self.check_layers}
        
        def return_uv(target_h, target_w, feat):
            hf, wf = feat.shape[2:]
            u = np.arange(wf)[None,:,None].repeat(hf,axis=0)
            v = np.arange(hf)[:,None,None].repeat(wf,axis=1)
            uv = np.concatenate([u, v],axis=-1) #32,44,2的矩阵，[[[0,0],[1,0]...][[0,1],[1,1]...]...]
            uv = uv.reshape(-1,2) #变为1408，2 [[0,0]...]
            uv = self.control.capturer.uv_back_to_origin(uv, target_h, target_w, hf, wf)
            return uv
    
        # uv = return_uv(target_h, target_w, feats[max(self.check_layers)]).reshape(-1,2)
        # return {'uv':uv, 'feat':feats}
        return feats
    
    def dpt_sd_feat(self, dpt):
        dpt_feats = self.control.dpt_feature(dpt)
        target_w = target_h = self.cfg.img_dim
        dpt_feats = {i:dpt_feats[i] for i in self.check_layers}
        def return_uv(target_h, target_w, feat):
            hf, wf = feat.shape[2:]
            u = np.arange(wf)[None,:,None].repeat(hf,axis=0)
            v = np.arange(hf)[:,None,None].repeat(wf,axis=1)
            uv = np.concatenate([u, v],axis=-1) #32,44,2的矩阵，[[[0,0],[1,0]...][[0,1],[1,1]...]...]
            uv = uv.reshape(-1,2) #变为1408，2 [[0,0]...]
            uv = self.control.capturer.uv_back_to_origin(uv, target_h, target_w, hf, wf)
            return uv

        uv = return_uv(target_h, target_w, dpt_feats[max(self.check_layers)]).reshape(-1,2)
        return {'uv':uv, 'feat':dpt_feats}
    
    def forward(self, batch):
        # Estimate Depth -- now for 1 and 2
        n_views = 2
        
        rgbs = [batch[f"rgb_{i}"] for i in range(n_views)]
        deps = [batch[f"depth_{i}"] for i in range(n_views)]
        
        rgb_sd_feats = [{j:batch[f"rgb_sd_{i}_{j}_list"] for j in range(4)} for i in range(n_views)]
        dep_sd_feats = [{j:batch[f"dep_sd_{i}_{j}_list"] for j in range(4)} for i in range(n_views)]

        K = batch["K"]
        
        output = {}
        B, _, H, W = rgbs[0].shape

        # Encode features
        feat_p_encode = []
        feat_i_encode = []

        #pre stage
        feat_p = batch['features'].clone().detach()
        for block_op in self.pcd_pre_stages: # pcd_pre_stages是一个modulelist：包含Simple Block(KPConv+BN+ReLU) 和 ResnetBottleneckBlock(downscaling+KPConv+bn+upcaling+unary_shortcut+relu)
            feat_p = block_op(feat_p, batch['points'], batch['neighbors'], batch['pools']) # pointcloud feature
        feat_p_origin = feat_p
                        
        if self.use_original:
            feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] #image feature [1,64,512,512]
        else:
        
            start_time = time.time()
            feat_img_0, feat_img_1 = self.fusion_attn(rgb_sd_feats, dep_sd_feats)
            # feat_img_0, feat_img_1 = self.feature_pca_and_aggregate(rgb_items, dep_items) # [B,64,64,64]
            end_time = time.time()
            # print(f"##### PCA NEED: {end_time - start_time} seconds#####")


        if not self.use_original:
            # feat_i = [feat_i_rgb[i].unsqueeze(0) + feat_i_dep[i].unsqueeze(0) for i in range(n_views)] # [B,64,64,64]
            # feat_i = [feat_img_0, feat_img_1]
            # feat_i_up = [F.interpolate(feat_i[i], scale_factor=4., mode='bilinear', align_corners=True) for i in range(n_views)] # [B,64,256,256]
            
            # img upsample
            
            start_time = time.time()
            feat_img_0_up = F.interpolate(feat_img_0, scale_factor=4., mode='bilinear', align_corners=True) # [B,64,512,512]
            feat_img_1_up = F.interpolate(feat_img_1, scale_factor=4., mode='bilinear', align_corners=True)
            
            # feat_i = [self.cnn_pre_stages(rgbs[i]) for i in range(n_views)] 

            # feat_img_0_up = feat_i[0]
            # feat_img_1_up = feat_i[1]
            
            # downsample 0 512->256
            feat_p2i = self.gather_p2i_origin(feat_p, batch['p2i_list'][0].squeeze(1)) # [B, XX, 128] -> [B, 2*512*512, 128]    # p2i_list记录的是在 完整点云中每个点 寻找 体素化的像素点的最近距离 # feat_p2i 是从体素化的点中挑出 完整点云最近点的 特征，成为了 
            feat_p2i = self.p2i_downsample_0[0](feat_p2i) # [B, 2*512*512, 128]  ->  [B, 2*512*512, 64]#对 完整点云每一个点对应的体素化的点云 进行下采样 128->64
            feat_p2i = self.fusep2i_origin([feat_img_0_up, feat_img_1_up], feat_p2i, self.p2i_downsample_0[1]) #[1,64,512,512]*2 [2*512*512, 64] -> [B,64,512,512]*2

            feat_i2p = self.gather_i2p_origin([feat_img_0_up, feat_img_1_up], batch['i2p_list'][0]) # [B,64,512,512]*2 -> [2, 19697, 16, 64]
            feat_i2p = self.i2p_downsample_0[0](feat_i2p.max(1)[0]) # [2, 19697, 16, 64] -> [2, 19697, 128]
            feat_i2p = self.fusei2p_origin(feat_p, feat_i2p, self.i2p_downsample_0[1]) # [2, 19697, 128] [2, 19697, 128] -> [2, 19697, 128]
            
            feat_p = feat_p + feat_i2p # [32962, 128] + [32962, 128] -> [32962, 128]
            feat_img_0 = feat_img_0_up + feat_p2i[0] # [B,128,256,256]
            feat_img_1 = feat_img_1_up + feat_p2i[1] # [B,128,256,256]

            feat_p_encode.append(feat_p)
            feat_i_encode.append(feat_img_0)
            feat_i_encode.append(feat_img_1)
            
            # downsample 1 256->128
            for block_op in self.pcd_ds_0:
                feat_p = block_op(feat_p, batch['points'], batch['neighbors'], batch['pools']) # [4677, 128] ->  [1101, 256]
            feat_img_0 = self.cnn_ds_0(feat_img_0_up) # [B,64,512,512] -> [B,128,64,64] 
            feat_img_1 = self.cnn_ds_0(feat_img_1_up) # [B,64,512,512] -> [B,128,64,64] 
            
            feat_p2i = self.gather_p2i_origin(feat_p, batch['p2i_list'][1].squeeze(1)) # [B, XXXX, 256] -> [2*256*256, 256] 
            feat_p2i = self.p2i_downsample_1[0](feat_p2i) # [2*256*256, 256] -> [B, 2*256*256, 128]
            feat_p2i = self.fusep2i_origin([feat_img_0, feat_img_1], feat_p2i, self.p2i_downsample_1[1])  # [1,128,256,256]*2, [2*256*256, 256] -> [B,128,256,256]*2
            
            feat_i2p = self.gather_i2p_origin([feat_img_0, feat_img_1], batch['i2p_list'][1]) # [1,128,256,256]*2 -> [1101,16,128]
            feat_i2p = self.i2p_downsample_1[0](feat_i2p.max(1)[0]) # [1101,16,128] -> [24186,256]
            feat_i2p = self.fusei2p_origin(feat_p, feat_i2p, self.i2p_downsample_1[1]) # [1101, 256], [1101,256] -> [1101,256]
            
            feat_p = feat_p + feat_i2p # [24186,256] + [24186,256] -> [24186,256]
            feat_img_0 = feat_img_0 + feat_p2i[0] # [B,128,64,64]
            feat_img_1 = feat_img_1 + feat_p2i[1] # [B,128,64,64]

            # upsample
            for block_i, block_op in enumerate(self.pcd_up_1):
                if block_i % 2 == 1:
                    feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
                feat_p = block_op(feat_p, batch['upsamples']) # [1101, 256] -> [4677, 32]

            feat_img_0 = self.cnn_up_1(torch.cat((F.interpolate(feat_img_0, scale_factor=2., mode='bilinear', align_corners=True), feat_img_0_up), dim=1)) #[1,128,512,512], [1,64,512,512] ->[1,32,512,512]
            feat_img_1 = self.cnn_up_1(torch.cat((F.interpolate(feat_img_1, scale_factor=2., mode='bilinear', align_corners=True), feat_img_1_up), dim=1))
            
            indices = torch.randperm(feat_img_0.size(2) * feat_img_0.size(3))[:500]
            pcs_X = batch['points_img']
            pcs_F = self.map(feat_img_0, feat_img_1, feat_p, batch['p2i_list'][0], indices) # [B,32,512,512]->[B,32,512*512], [4677,32] ->p2i-> [B,32,512,512] ====>> [B, 512*512, 32]
            end_time = time.time()
            
            pcs_X[0] = pcs_X[0][:, indices, :]
            pcs_X[1] = pcs_X[1][:, indices, :]
            
            # print(f"##### NET NEED: {end_time - start_time} seconds#####")
            
            # box embedding
        else:
        
        
        
        
            
            #***************#
            
            feat_p2i = self.gather_p2i_origin(feat_p, batch['p2i_list'][0].squeeze()) # [4677, 128] -> [2*512*512, 128]    # p2i_list记录的是在 完整点云中每个点 寻找 体素化的像素点的最近距离 # feat_p2i 是从体素化的点中挑出 完整点云最近点的 特征，成为了 
            feat_p2i = self.fuse_p2i_ds_0[0](feat_p2i) # [2*512*512, 128]  ->  [2*512*512, 64]#对 完整点云每一个点对应的体素化的点云 进行下采样 128->64
            feat_p2i = self.fusep2i_origin(feat_i, feat_p2i, self.fuse_p2i_ds_0[1]) #[1,64,512,512]*2 [2*512*512, 64] -> [1,64,512,512]*2

            feat_i2p = self.gather_i2p_origin(feat_i, batch['i2p_list'][0]) # [1,64,512,512]*2 -> [4677, 16, 64]
            feat_i2p = self.fuse_i2p_ds_0[0](feat_i2p.max(1)[0]) # [4677, 16, 64] -> [4677, 128]
            feat_i2p = self.fusei2p_origin(feat_p, feat_i2p, self.fuse_i2p_ds_0[1]) # [4677, 128] [4677, 128] -> [4677, 128]

            feat_p = feat_p + feat_i2p # [4677, 128] + [4677, 128] -> [4677, 128]
            feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)] # [1,64,512,512] + [1,64,512,512] -> [1,64,512,512]

            feat_p_encode.append(feat_p)
            feat_i_encode.append(feat_i)

            #downsample 0
            for block_op in self.pcd_ds_0:
                feat_p = block_op(feat_p, batch)
            feat_i = [self.cnn_ds_0(feat_i[i]) for i in range(n_views)]

            feat_p2i = self.gather_p2i_origin(feat_p, batch['p2i_list'][1].squeeze())
            feat_p2i = self.fuse_p2i_ds_1[0](feat_p2i)
            feat_p2i = self.fusep2i_origin(feat_i, feat_p2i, self.fuse_p2i_ds_1[1])

            feat_i2p = self.gather_i2p_origin(feat_i, batch['i2p_list'][1]) #  [1,128,256,256]*2 -> [1101,16,128]
            feat_i2p = self.fuse_i2p_ds_1[0](feat_i2p.max(1)[0]) # [1101,16,128] -> [1101,256]
            feat_i2p = self.fusei2p_origin(feat_p, feat_i2p, self.fuse_i2p_ds_1[1]) # [1101,256], [1101,256]

            feat_p = feat_p + feat_i2p
            feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

            feat_p_encode.append(feat_p)
            feat_i_encode.append(feat_i)

            #downsample 1
            for block_op in self.pcd_ds_1:
                feat_p = block_op(feat_p, batch)
            feat_i = [self.cnn_ds_1(feat_i[i]) for i in range(n_views)]

            feat_p2i = self.gather_p2i_origin(feat_p, batch['p2i_list'][2].squeeze())
            feat_p2i = self.fuse_p2i_ds_2[0](feat_p2i)
            feat_p2i = self.fusep2i_origin(feat_i, feat_p2i, self.fuse_p2i_ds_2[1])

            feat_i2p = self.gather_i2p_origin(feat_i, batch['i2p_list'][2])
            feat_i2p = self.fuse_i2p_ds_2[0](feat_i2p.max(1)[0])
            feat_i2p = self.fusei2p_origin(feat_p, feat_i2p, self.fuse_i2p_ds_2[1])

            feat_p = feat_p + feat_i2p
            feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]


            # upsample0
            for block_i, block_op in enumerate(self.pcd_up_0):
                if block_i % 2 == 1:
                    feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
                feat_p = block_op(feat_p, batch)

            feat_i = [
                self.cnn_up_0(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                        feat_i_encode[-1][i]), dim=1)) for i in range(n_views)] #interpolate上采样插值

            feat_p2i = self.gather_p2i(feat_p, batch['p2i_list'][1].squeeze())
            feat_p2i = self.fuse_p2i_up_0[0](feat_p2i)
            feat_p2i = self.fusep2i(feat_i, feat_p2i, self.fuse_p2i_up_0[1])

            feat_i2p = self.gather_i2p(feat_i, batch['i2p_list'][1])
            feat_i2p = self.fuse_i2p_up_0[0](feat_i2p.max(1)[0])
            feat_i2p = self.fusei2p(feat_p, feat_i2p, self.fuse_i2p_up_0[1])

            feat_p = feat_p + feat_i2p
            feat_i = [feat_i[i] + feat_p2i[i] for i in range(n_views)]

            # upsample1
            for block_i, block_op in enumerate(self.pcd_up_1):
                if block_i % 2 == 1:
                    feat_p = torch.cat([feat_p, feat_p_encode.pop()], dim=1)
                feat_p = block_op(feat_p, batch)

            feat_i = [
                self.cnn_up_1(torch.cat((F.interpolate(feat_i[i], scale_factor=2., mode='bilinear', align_corners=True),
                                        feat_i_encode[-2][i]), dim=1)) for i in range(n_views)]

            pcs_X = batch['points_img']
            pcs_F = self.map(feat_i, feat_p, batch) # [1,32,512,512]->[1,32,512*512], [4677,32] ->p2i-> [1,32,512,512] ====>> [1, 512*512, 32]
        
        if self.align_cfg.algorithm == "weighted_procrustes":
            vps = []
            cor_loss = []
            for i in range(1, n_views):
                
                start_time = time.time()
                m12_idx1, m12_idx2, m12_dist = get_correspondences_1(
                    self.use_original,
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                    gumbel_beta = 0.0036026463511690845
                )
                end_time = time.time()
                # print(f"##### EACH CORR + BOX NEED: {end_time - start_time} seconds#####")
                
                m21_idx2, m21_idx1, m21_dist = get_correspondences_2(
                    self.use_original,
                    P1=pcs_F[0],
                    P2=pcs_F[i],
                    P1_X=pcs_X[0],
                    P2_X=pcs_X[i],
                    num_corres=self.num_corres,
                    ratio_test=(self.align_cfg.base_weight == "nn_ratio"),
                    gumbel_beta = 0.0036026463511690845
                )
                """
                # P1          FloatTensor (N x P x F)     features for first pointcloud
                # P2          FloatTensor (N x Q x F)     features for second pointcloud
                # num_corres  Int                         number of correspondances
                # P1_X        FloatTensor (N x P x 3)     xyz for first pointcloud
                # P2_X        FloatTensor (N x Q x 3)     xyz for second pointcloud
                # metric      {cosine, euclidean}         metric to be used for kNN
                # ratio_test  Boolean                     whether to use ratio test for kNN
                """
                matches_idx1 = torch.cat((m12_idx1, m21_idx1), dim=1).squeeze(dim=2)
                matches_idx2 = torch.cat((m12_idx2, m21_idx2), dim=1).squeeze(dim=2)
                matches_dist = torch.cat((m12_dist, m21_dist), dim=1).squeeze(dim=2)
                corr_i = [matches_idx1, matches_idx2, matches_dist]
                
                Rt_i, cor_loss_i = align(corr_i, pcs_X[0], pcs_X[i], self.align_cfg)

                vps.append(Rt_i)
                cor_loss.append(cor_loss_i)

                # add for visualization
                output[f"corres_0{i}"] = corr_i
                output[f"vp_{i}"] = Rt_i
        else:
            raise ValueError(f"How to align using {self.align_cfg.algorithm}?")

        # add correspondance loss to output
        output["corr_loss"] = sum(cor_loss)
        
        # Rotate points into the frame of the view image
        pcs_X_rot = [
            transform_points_Rt(pcs_X[i + 1], vps[i], inverse=True)
            for i in range(n_views - 1)
        ]
        pcs_X = pcs_X[0:1] + pcs_X_rot
        output["joint_pointcloud"] = torch.cat(pcs_X, dim=1).detach().cpu()

        # Get RGB pointcloud as well for direct rendering
        pcs_rgb = [rgb.view(B, 3, -1).permute(0, 2, 1).contiguous() for rgb in rgbs]

        projs = []
        # get joint for all values
        if self.pointcloud_source == "joint":
            pcs_X_joint = torch.cat(pcs_X, dim=1)
            pcs_F_joint = torch.cat(pcs_F, dim=1)
            pcs_RGB_joint = torch.cat(pcs_rgb, dim=1)
            pcs_FRGB_joint = torch.cat((pcs_F_joint, pcs_RGB_joint), dim=2)

        # Rasterize and Blend
        for i in range(n_views):
            if self.pointcloud_source == "other":
                # get joint for all values except the one
                pcs_X_joint = torch.cat(pcs_X[0:i] + pcs_X[i + 1: n_views], dim=1)
                pcs_center_joint = pcs_F[i].center
                pcs_min_offset_joint = pcs_F[i].min_offset
                pcs_max_offset_joint = pcs_F[i].max_offset
                pcs_RGB_joint = torch.cat(
                    pcs_rgb[0:i] + pcs_rgb[i + 1: n_views], dim=1
                )
                pcs_FRGB_joint = torch.cat((pcs_center_joint, pcs_min_offset_joint, pcs_max_offset_joint, pcs_RGB_joint[:, indices, :]), dim=2)
              
            if i > 0:
                rot_joint_X = transform_points_Rt(pcs_X_joint, vps[i - 1])
                rot_joint_X = points_to_ndc(rot_joint_X, K, (H, W))
            else:
                rot_joint_X = points_to_ndc(pcs_X_joint.float(), K, (H, W))
            projs.append(self.renderer(rot_joint_X, pcs_FRGB_joint)) # pcs_FRGB_joint # points features  光栅化是将三维空间中的图形或对象投影到二维平面上的过程

        # Decode
        for i in range(n_views):
            proj_FRGB_i = projs[i]["feats"]
            proj_RGB_i = proj_FRGB_i[:, -3:]
            proj_F_i = proj_FRGB_i[:, :-3]

            output[f"rgb_render_{i}"] = proj_RGB_i
            output[f"ras_depth_{i}"] = projs[i]["depth"]
            output[f"cover_{i}"] = projs[i]["mask"].unsqueeze(1)  # useless
            output[f"indices"] = indices

        return output

    def gather_p2i(self, feat_p, idx):
        B, _, _ = feat_p.shape
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:, :1, :])), 1)
        return torch.stack([feat_p[:, idx][i][i] for i in range(B)])
    

    def gather_i2p(self, src_feat_i, tgt_feat_i, idx):
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))
        # feat_i2p.append(src_feat_i.reshape(B,C,H*W).permute(0,2,1))
        # feat_i2p.append(tgt_feat_i.reshape(B,C,H*W).permute(0,2,1))
        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C
    
    def gather_p2i_origin(self, feat_p, idx):
        feat_p = torch.cat((feat_p, torch.zeros_like(feat_p[:1, :])), 0)
        return feat_p[idx]

    def gather_i2p_origin(self, feat_i, idx):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        feat_i2p = []
        for i in range(B):
            feat_i2p.append(src_feat_i[i].reshape(C,H*W).permute(1,0))
            feat_i2p.append(tgt_feat_i[i].reshape(C,H*W).permute(1,0))

        feat_i2p = torch.cat(feat_i2p, 0)
        feat_i2p = torch.cat((feat_i2p, torch.zeros_like(feat_i2p[:1, :])), 0)
        return feat_i2p[idx]  #N,16,C
    
    def fusei2p_origin(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p
    
    def fusep2i_origin(self, feat_i, feat_p2i, layer):
        src_feat_i, tgt_feat_i = feat_i
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []

        for i in range(2*B):
            if i % 2 == 0:
                src_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                tgt_feat_p2i.append(feat_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]
    
    def fusep2i(self, src_feat_i, tgt_feat_i, feat_p2i, layer):
        B, C, H, W = src_feat_i.shape
        # src_feat_i = src_feat_i.reshape(B, C, H * W).permute(0, 2, 1)
        # tgt_feat_i = tgt_feat_i.reshape(B, C, H * W).permute(0, 2, 1)

        src_feat_p2i = []
        tgt_feat_p2i = []
        
        for i in range(B):
            src_feat_p2i.append(feat_p2i[i, 0*H*W : 1*H*W].unsqueeze(0))
            tgt_feat_p2i.append(feat_p2i[i, 1*H*W : 2*H*W].unsqueeze(0))

        src_feat_p2i = torch.vstack(src_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)
        tgt_feat_p2i = torch.vstack(tgt_feat_p2i).permute(0, 2, 1).reshape(B, C, H, W)

        src_feat_p2i = torch.cat([src_feat_i, src_feat_p2i], 1)
        tgt_feat_p2i = torch.cat([tgt_feat_i, tgt_feat_p2i], 1)

        src_feat_p2i = layer(src_feat_p2i)
        tgt_feat_p2i = layer(tgt_feat_p2i)

        return [src_feat_p2i, tgt_feat_p2i]

    def fusei2p(self, feat_p, feat_i2p, layer):
        feat_i2p = torch.cat([feat_p, feat_i2p], -1)
        feat_i2p = layer(feat_i2p)
        return feat_i2p


    def generate_pointclouds(self, K, deps, vps=None):
        n_views = len(deps)
        # generate pointclouds - generate grid once for efficiency
        B, _, H, W = deps[0].shape
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pcs_X = [
            grid_to_pointcloud(K_inv, deps[i], None, grid)[0] for i in range(n_views)
        ]

        if vps is not None:
            pcs_X_rot = [
                transform_points_Rt(pcs_X[i + 1], vps[i + 1], inverse=True, )
                for i in range(n_views - 1)
            ]
            pcs_X = pcs_X[0:1] + pcs_X_rot
            pcs_X = torch.cat(pcs_X, dim=1).detach().cpu()

        return pcs_X

    def get_feature_pcs(self, rgbs, K, deps):
        # Estimate Depth -- now for 1 and 2
        n_views = len(rgbs)

        # Encode features
        feats = [self.encode(rgbs[i]) for i in range(n_views)]

        # generate pointclouds - generate grid once for efficience
        B, _, H, W = feats[0].shape
        assert (
                feats[0].shape[-1] == deps[0].shape[-1]
        ), f"Same size {feats[0].shape} - {deps[0].shape}"
        grid = get_grid(B, H, W)
        grid = grid.to(deps[0])

        K_inv = K.inverse()
        pointclouds = [
            grid_to_pointcloud(K_inv, deps[i], feats[i], grid) for i in range(n_views)
        ]
        pcs_X = [pc[0] for pc in pointclouds]
        pcs_F = [pc[1] for pc in pointclouds]
        return pcs_X, pcs_F, None






