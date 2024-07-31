import torch
import einops
import numpy as np
from PIL import Image
from copy import deepcopy
from .capture import capture
from .utils import resize_image, HWC3
from .cldm.ddim_hacked import DDIMSampler

class null_img_processor():
    def sd_single_img(self, img_fn, prompt):
        return img_fn, prompt

class null_dpt_processor():
    def control_single_dpt(self, dpt_fn, prompt, dpt):
        return dpt_fn, prompt

class Null_model():  
    def __init__(self) -> None:
        self.prompt = ''
        self.capturer = capture(load_model=False)
        self.img_processing = null_img_processor()
        self.dpt_processing = null_dpt_processor()

class control_extractor(Null_model):
    def __init__(self, 
                 cfg,
                 load_model = True,
                 seed = -1, 
                 t = 150, 
                 basic = './tools/controlnet/models', 
                 yaml = 'control_v11f1p_sd15_depth.yaml', 
                 sd_ckpt = 'v1-5-pruned.ckpt', 
                 cn_ckpt = 'control_v11f1p_sd15_depth.pth', 
                 prompt = 'a photo of a room and furniture',
                 ) -> None:
        super().__init__()
        if load_model:
            self.prompt = prompt
            self.capturer = capture(seed = seed, basic = basic, 
                                    yaml = yaml, sd_ckpt = sd_ckpt,
                                    cn_ckpt = cn_ckpt, t = t)
            self.cfg = cfg
            self.img_processing = img_processor(self.capturer, self.cfg)
            self.dpt_processing = dpt_processor(self.capturer, self.cfg)

    def dpt_feature(self, dpt_fn = '', dpt = None):
        feat_list = self.dpt_processing.control_single_dpt(dpt_fn = dpt_fn, prompt = self.prompt, dpt = dpt)
        return feat_list

    def rgb_feature(self, img_fn):
        feat_list = self.img_processing.sd_single_img(img_fn, prompt = self.prompt)
        return feat_list

class img_processor():
    def __init__(self, 
                 capturer,
                 cfg,
                 ) -> None:
        # basic
        self.capturer = capturer
        self.check_layers = [0,4,6,11]
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_rgb(self, rgb_fn):
        rgb = np.array(Image.open(rgb_fn)).astype(np.uint8) #0-255
        img = deepcopy(rgb)
        img = HWC3(img)
        img = resize_image(img, self.capturer.img_resolution)
        return rgb, img
    
    def add_noise(self, img):
        # input img(feat_map) should be b*c*h*w
        img = self.capturer.model.q_sample(img, self.capturer.tlist)
        return img

    def sd_input(self, img):
        # rgb loading
        self.H, self.W, self.C = img.shape
        img = (torch.from_numpy(np.array(img).astype(np.float32))-127.5)/ 127.5  # must be [-1,1]
        img = einops.rearrange(img[None], 'b h w c -> b c h w').clone()
        img = img.to(self.device)
        return img

    def sd_process(self, img, prompt = ''):
        # diffusion encoding
        img = self.capturer.model.encode_first_stage(img)
        # encoder comes f -> batch*4*64*64
        img = self.capturer.model.get_first_stage_encoding(img).detach() 
        # add noise
        noise_img = self.add_noise(img)
        # diffusion u-net
        cond = {"c_crossattn": [self.capturer.model.get_learned_conditioning([prompt + ', ' + self.capturer.a_prompt])]}
        cond_txt = torch.cat(cond['c_crossattn'], 1).to(self.device)
        cond_txt = cond_txt.expand(self.cfg.batch_size, -1, -1)
        with torch.no_grad():
            _, inter_feats = self.capturer.model.model.diffusion_model(x=noise_img, 
                                timesteps=self.capturer.tlist, 
                                context=cond_txt, 
                                control=None, 
                                only_mid_control=self.capturer.only_mid_control,
                                per_layers = True)
        # 1*c*h*w -> c*h*w
        # inter_feats = [i[0].detach() for i in inter_feats]
        return inter_feats

    def sd_single_img(self, img_fn:str, prompt = ''):
        # sd input:
        # rgb, img = self.load_rgb(img_fn)
        # diffusion features
        # img = self.sd_input(img)
        # a list of c*h*w layer feature maps at self.capturer.t
        chw_feat_list = self.sd_process(img_fn, prompt=prompt)
        return chw_feat_list

    def img_match(self, img_fn_source, img_fn_target, check_layers, pca_dim = 256):
        # get features
        rgbs, featlist_s = self.sd_single_img(img_fn_source)
        rgbt, featlist_t = self.sd_single_img(img_fn_target)
        # feature list to feature map
        fs, ft = self.capturer.merge_feat(featlist_s, featlist_t, check_layers, pca_dim = pca_dim)
        # conduct feature match
        uvs, uvt = self.capturer.chw_img_match(fs, ft)
        # uvst back to origin image size
        uvs = self.capturer.uv_back_to_origin(uvs, rgbs.shape[0], rgbs.shape[1], fs.shape[1], fs.shape[2])
        uvt = self.capturer.uv_back_to_origin(uvt, rgbt.shape[0], rgbt.shape[1], ft.shape[1], ft.shape[2])
        # draw
        self.capturer.draw_match(rgbs, rgbt, uvs, uvt)
        
        
class dpt_processor():
    def __init__(self, 
                 capturer,
                 cfg,
                 ) -> None:
        # basic
        self.capturer = capturer
        self.ddim_sampler = DDIMSampler(self.capturer.model)
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def depth_normalize(self, depth):
        # following controlnet  1-depth
        depth = depth.astype(np.float64)
        vmin = np.percentile(depth, 2)
        vmax = np.percentile(depth, 85)
        depth -= vmin
        depth /= vmax - vmin
        depth = 1.0 - depth
        depth_image = (depth * 255.0).clip(0, 255).astype(np.uint8)
        return depth_image

    def process_given_dpt(self, dpt_backup):
        dpt = deepcopy(dpt_backup)
        # depth normalization -> 0-255 uint8
        dpt = self.depth_normalize(dpt)
        # dpt as network input
        dpt = HWC3(dpt)
        # dpt = cv2.resize(dpt, self.capturer.img_resolution, interpolation=cv2.INTER_LINEAR) 
        dpt = resize_image(dpt, self.capturer.img_resolution) 
        dpt = np.array(dpt)
        self.H, self.W = dpt.shape[0:2]
        # for visualization
        dpt_backup = dpt_backup[:,:,None].repeat(3,axis=-1).astype(np.float32)
        return dpt_backup, dpt
    
    def control_process(self, dpt, prompt = '', final_output = False):
        # noise input -> sd decoding
        cond = {"c_concat": [dpt], 
                "c_crossattn": [self.capturer.model.get_learned_conditioning([prompt + ', ' + self.capturer.a_prompt])]}
        un_cond = {"c_concat": None if self.capturer.guess_mode else [dpt], 
                   "c_crossattn": [self.capturer.model.get_learned_conditioning([self.capturer.n_prompt])]}
        cond_txt = torch.cat(cond['c_crossattn'], 1).to(self.device)
        cond_txt = cond_txt.expand(self.cfg.batch_size, -1, -1)
        shape = (4, self.cfg.img_dim // 8, self.cfg.img_dim // 8)
        # conduct diffusion to step 100
        _, intermediates = self.ddim_sampler.sample(self.capturer.steps,          # how many diffusion steps
                                                     self.cfg.batch_size,                           # generate how many results, we need 1 only
                                                     shape,                       # to explain
                                                     cond,                        # depth, prompts
                                                     verbose=False,                         
                                                     eta=self.capturer.eta,
                                                     unconditional_guidance_scale=self.capturer.uncond_scale,     
                                                     unconditional_conditioning=un_cond,
                                                     log_every_t=1)    # with depth guidance and generate unconvincing results  -- should not be
        steps = intermediates['step']
        intermediates = intermediates['x_inter']
        # the t-th iteration
        index = round((1000 - self.capturer.t)/(1000/self.capturer.steps))
        render = intermediates[index]
        step = steps[index]
        x_samples = self.capturer.model.decode_first_stage(intermediates[-1])

        control_model = self.capturer.model.control_model
        diffusion_model = self.capturer.model.model.diffusion_model
        control = control_model(x=render, 
                                hint=torch.cat(cond['c_concat'], 1), 
                                timesteps=self.capturer.tlist, 
                                context=cond_txt)
        control = [c * scale for c, scale in zip(control, self.capturer.control_scales)]
        with torch.no_grad():
            _, inter_feats = diffusion_model(x=render, 
                                timesteps=self.capturer.tlist, 
                                context=cond_txt, 
                                control=control, 
                                only_mid_control=self.capturer.only_mid_control,
                                per_layers = True)
        # 1*c*h*w -> c*h*w
        # inter_feats = [i[0].detach() for i in inter_feats]
        if final_output:
            return inter_feats, x_samples
        return inter_feats
    
    def control_single_dpt(self, dpt_fn = '', prompt = '', dpt = None):
        # # load dpt
        # if dpt is None:
        #     depth, dpt = self.load_dpt(dpt_fn)
        # else:
        #     depth, dpt = self.process_given_dpt(dpt)
        # control input
        dpt = torch.cat([dpt_fn] * 3, dim=1)
        # process
        chw_feat_list = self.control_process(dpt,prompt=prompt)
        return chw_feat_list