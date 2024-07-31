import torch
import torch.nn as nn
from typing import TypeVar
from torch import Tensor
from typing import Type
import gc 
import open3d as o3d
from einops import rearrange
Hyper_Embed = TypeVar("Hyper_Embed", bound="Hyper_Embedding")

def _box_shape_ok(t1: Tensor, t2: Tensor) -> bool:
    if len(t1.shape) < 2 or len(t2.shape) < 2:
        return False
    else:
        if t2.size(-2) != 2:
            return False
        return True

class Hyper_Embedding(nn.Module):
    def __init__(self, center: Tensor, offset: Tensor) -> None:
        super().__init__()
        if _box_shape_ok(center, offset):
            self._center = nn.Parameter(center)
            self._offset = nn.Parameter(offset)
        else:
            error_message = (
                            _shape_error_str('center', '(**,2,num_dims)', center.shape) + '\n' +
                            _shape_error_str('offset', '(**,2,num_dims)', offset.shape)
                        )
            raise ValueError(error_message)
    
    def __getitem__(self, index) -> Hyper_Embed:
        return Hyper_Embedding(self._center[:,index,:], self._offset[index,:,:])

    @property
    def center(self) -> Tensor:
        """Centre coordinate as Tensor"""
        return self._center

    @property
    def min_offset(self) -> Tensor:
        """Lower left coordinate as Tensor"""

        return self._offset[..., 0, :]

    @property
    def max_offset(self) -> Tensor:
        """Top right coordinate as Tensor"""
        return self._offset[..., 1, :]
    
    @property
    def gumbel_center(self) -> Tensor:
        return self._center

    @property
    def gumbel_min_offset(self) -> Tensor:
        min_offset = self._offset[..., 0, :] \
        - torch.nn.functional.softplus(self._offset[..., 1, :], beta=10.)
        return torch.sigmoid(min_offset)

    @property
    def gumbel_max_offset(self) -> Tensor:
        max_offset = self._offset[..., 0, :] \
        + torch.nn.functional.softplus(self._offset[..., 1, :], beta=10.)
        return torch.sigmoid(max_offset)
    
    @classmethod
    def from_zZ(cls: Type[Hyper_Embed], center: Tensor, min_offset: Tensor, max_offset: Tensor) -> Hyper_Embed:
        
        if min_offset.shape != max_offset.shape:
            raise ValueError(
                "Shape of z and Z should be same but is {} and {}".format(
                min_offset.shape, max_offset.shape))
        offset: Tensor = torch.stack((min_offset, max_offset), -2)

        return cls(center, offset)
    
    @classmethod
    def from_split(cls: Type[Hyper_Embed], t: Tensor,
            dim: int = -1) -> Hyper_Embed:
        """Creates a BoxTensor by splitting on the dimension dim at midpoint

        Args:
        t: input
        dim: dimension to split on

        Returns:
        BoxTensor: output BoxTensor

        Raises:
        ValueError: `dim` has to be even
        """
        len_dim = t.size(dim)

        if len_dim % 2 != 0:
            raise ValueError(
                "dim has to be even to split on it but is {}".format(
                t.size(dim)))
            split_point = int(len_dim / 2)
            t.to('cuda')
            min_offset = t.index_select(dim, torch.tensor(list(range(split_point)), dtype=torch.int64, device=t.device))

            max_offset = t.index_select(dim,torch.tensor(list(range(split_point, len_dim)), dtype=torch.int64, device=t.device))

        return cls.from_zZ(min_offset, max_offset)
    
class Gumbel_Hyper_Embedding(Hyper_Embedding):
    
    def __init__(self, center: Tensor, offset: Tensor) -> None:
        super().__init__()
        self.center = center
        self.offset = offset
        
    @property
    def gumbel_center(self) -> Tensor:
        return self.center

    @property
    def gumbel_min_offset(self) -> Tensor:
        min_offset = self.offset[0, :] \
        - torch.nn.functional.softplus(self.offset[1, :], beta=10.)
        return torch.sigmoid(min_offset)

    @property
    def gumbel_max_offset(self) -> Tensor:
        max_offset = self.offset[0, :] \
        + torch.nn.functional.softplus(self.offset[1, :], beta=10.)
        return torch.sigmoid(max_offset)
    
    @classmethod
    def gumbel_intersection_box(self, box1, box2, gumbel_beta):
        intersections_min = gumbel_beta * torch.logsumexp(
                torch.stack((box1.gumbel_min_offset / gumbel_beta, box2.gumbel_min_offset / gumbel_beta)),
                0
        )
        intersections_min = torch.max(
            intersections_min,
            torch.max(box1.gumbel_min_offset, box2.gumbel_min_offset)
        )
        intersections_max = - gumbel_beta * torch.logsumexp(
            torch.stack((-box1.gumbel_max_offset / gumbel_beta, -box2.gumbel_max_offset / gumbel_beta)),
            0
        )
        intersections_max = torch.min(
            intersections_max,
            torch.min(box1.gumbel_max_offset, box2.gumbel_max_offset)
        )
        
        return intersections_min, intersections_max
    
    # box1s center [1,262144,256] offset [262144,2,128]; box2s center [1,262144,3,256] offset [262144,3,2,128]
    @classmethod
    def gumbel_intersection_boxs_batch(self, box1s_min, box1s_max, box2s_min, box2s_max, gumbel_beta):
        if len(box2s_min.size()) == 4:
            knn_point_num = box2s_min.shape[2]
            intersections_mins = gumbel_beta * torch.logsumexp(
                torch.stack((box1s_min.unsqueeze(2).expand(-1,-1,knn_point_num,-1) / gumbel_beta, box2s_min / gumbel_beta)),
                0
            )
            intersections_mins = torch.max(
                intersections_mins,
                torch.max(box1s_min.unsqueeze(2).expand(-1,-1,knn_point_num,-1), box2s_min)
            )
            intersections_maxs = - gumbel_beta * torch.logsumexp(
                torch.stack((-box1s_max.unsqueeze(2).expand(-1,-1,knn_point_num,-1) / gumbel_beta, -box2s_max / gumbel_beta)),
                0
            )
            intersections_maxs = torch.min(
                intersections_maxs,
                torch.min(box1s_max.unsqueeze(2).expand(-1,-1,knn_point_num,-1), box2s_max)
            )

        else:
            assert len(box1s_min.size()) == 4
            knn_point_num = box1s_min.shape[2]
            intersections_mins = gumbel_beta * torch.logsumexp(
                torch.stack((box1s_min / gumbel_beta, box2s_min.unsqueeze(2).expand(-1,-1,knn_point_num,-1) / gumbel_beta)),
                0
            )
            intersections_mins = torch.max(
                intersections_mins,
                torch.max(box1s_min, box2s_min.unsqueeze(2).expand(-1,-1,knn_point_num,-1))
            )
            intersections_maxs = - gumbel_beta * torch.logsumexp(
                torch.stack((-box1s_max / gumbel_beta, -box2s_max.unsqueeze(2).expand(-1,-1,knn_point_num,-1) / gumbel_beta)),
                0
            )
            intersections_maxs = torch.min(
                intersections_maxs,
                torch.min(box1s_max, box2s_max.unsqueeze(2).expand(-1,-1,knn_point_num,-1))
            )

        return intersections_mins, intersections_maxs
    
    @classmethod
    def gumbel_intersection_boxs(self, box1s_min, box1s_max, box2s_min, box2s_max, gumbel_beta):
        if len(box2s_min.size()) == 3:
            knn_point_num = box2s_min.shape[1]
            intersections_mins = gumbel_beta * torch.logsumexp(
                torch.stack((box1s_min.unsqueeze(1).expand(-1,knn_point_num,-1) / gumbel_beta, box2s_min / gumbel_beta)),
                0
            )
            intersections_mins = torch.max(
                intersections_mins,
                torch.max(box1s_min.unsqueeze(1).expand(-1,knn_point_num,-1), box2s_min)
            )
            intersections_maxs = - gumbel_beta * torch.logsumexp(
                torch.stack((-box1s_max.unsqueeze(1).expand(-1,knn_point_num,-1) / gumbel_beta, -box2s_max / gumbel_beta)),
                0
            )
            intersections_maxs = torch.min(
                intersections_maxs,
                torch.min(box1s_max.unsqueeze(1).expand(-1,knn_point_num,-1), box2s_max)
            )

        else:
            assert len(box1s_min.size()) == 3
            knn_point_num = box1s_min.shape[1]
            intersections_mins = gumbel_beta * torch.logsumexp(
                torch.stack((box1s_min / gumbel_beta, box2s_min.unsqueeze(1).expand(-1,knn_point_num,-1) / gumbel_beta)),
                0
            )
            intersections_mins = torch.max(
                intersections_mins,
                torch.max(box1s_min, box2s_min.unsqueeze(1).expand(-1,knn_point_num,-1))
            )
            intersections_maxs = - gumbel_beta * torch.logsumexp(
                torch.stack((-box1s_max / gumbel_beta, -box2s_max.unsqueeze(1).expand(-1,knn_point_num,-1) / gumbel_beta)),
                0
            )
            intersections_maxs = torch.min(
                intersections_maxs,
                torch.min(box1s_max, box2s_max.unsqueeze(1).expand(-1,knn_point_num,-1))
            )

        return [intersections_mins, intersections_maxs]
    
    @classmethod
    def log_soft_volume(self, min_offset, max_offset, euler_gamma, temp: float = 1.,scale: float = 1.,gumbel_beta: float = 0.):
        eps = torch.finfo(min_offset.dtype).tiny  # type: ignore

        if isinstance(scale, float):
            s = torch.tensor(scale)
        else:
            s = scale
        if gumbel_beta <= 0.:
            log_vol = torch.sum(torch.log(torch.nn.functional.softplus(max_offset - min_offset, beta=temp).clamp_min(eps)), dim=-1) + torch.exp(s)
            return log_vol              # need this eps to that the derivative of log does not blow
        else:
            # diff = max_offset - min_offset - 2 * euler_gamma * gumbel_beta
            # del max_offset, min_offset
            # gc.collect()
            
            # softplus_output = torch.nn.functional.softplus(max_offset - min_offset - 2 * euler_gamma * gumbel_beta, beta=temp).clamp_min(eps)
            # del diff
            # gc.collect()
            
            # log_vol = torch.sum(torch.log(softplus_output), dim=-1) + torch.exp(s)
            # del softplus_output
            # gc.collect()
            
            
            log_vol = torch.sum(torch.log(torch.nn.functional.softplus(max_offset - min_offset - 2 * euler_gamma * gumbel_beta, beta=temp).clamp_min(eps)), dim=-1) + torch.exp(s)
            
            return log_vol
        
    @classmethod
    def log_soft_mutli_volume(self, min_offsets, max_offsets, euler_gamma, temp: float = 1.,scale: float = 1.,gumbel_beta: float = 0.):
        box_num = min_offsets.shape[0]
        for i in range(box_num):
            log_vol = self.log_soft_volume(min_offsets[i], max_offsets[i], euler_gamma, temp, scale, gumbel_beta)
        if i == 0:
            log_vols = log_vol.unsqueeze(0)
        else:
            log_vols = torch.cat((log_vols, log_vol.unsqueeze(0)), 0)
        return log_vols
    
class Hyper_Emb_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, alpha = 0.1):
        super(Hyper_Emb_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        
    def voxel_downsample(point, voxel_size):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point.numpy())
        point_cloud = point_cloud.voxel_down_sample(voxel_size)
        return torch.from_numpy(np.array(point_cloud.points))
    
    def forward(self, feat_img_0, feat_img_1, feats_P, p2i_list_0, indices):
        B, C, H, W = feat_img_0.shape
        feats_I_src = feat_img_0.reshape(B, C, H*W)
        feats_I_tgt = feat_img_1.reshape(B, C, H*W)
        
        feats_I_src = self.fc(feats_I_src.permute(0, 2, 1))
        feats_I_tgt = self.fc(feats_I_tgt.permute(0, 2, 1))
        
        feats_P = torch.cat((feats_P, torch.zeros_like(feats_P[:1, :])), 0)
        feats_p2i = feats_P[p2i_list_0.squeeze()]
        # feats_p2i = torch.stack([feats_P[:, p2i_list_0.squeeze(2)][i][i] for i in range(B)])
        
        feats_p_src = []
        feats_p_tgt = []
        
        feats_p2i_src = []
        feats_p2i_tgt = []
        
        for i in range(2*B):
            if i%2 == 0:
                feats_p2i_src.append(feats_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
            else:
                feats_p2i_tgt.append(feats_p2i[i*H*W: (i+1)*H*W].unsqueeze(0))
                
        for i in range(B):
            feats_p_src.append(torch.cat((feats_I_src[i].unsqueeze(0), self.fc(feats_p2i_src[i])), dim = 2))
            feats_p_tgt.append(torch.cat((feats_I_tgt[i].unsqueeze(0), self.fc(feats_p2i_tgt[i])), dim = 2))
        
        feats_p_src = torch.cat(feats_p_src, dim = 0).to("cuda")
        feats_p_tgt = torch.cat(feats_p_tgt, dim = 0).to("cuda")
        
        
    
        # 使用索引序列选择样本
        sampled_feats_I_src = feats_I_src[:, indices, :]
        sampled_feats_I_tgt = feats_I_tgt[:, indices, :]
        sampled_feats_p_src = feats_p_src[:, indices, :]
        sampled_feats_p_tgt = feats_p_tgt[:, indices, :]
        
        
        left_bottom_src = sampled_feats_p_src[..., :, : self.output_dim] # [512*512,128]
        right_top_src = sampled_feats_p_src[..., :, self.output_dim:] # [512*512,128]
        emb_src = Hyper_Embedding.from_zZ(sampled_feats_I_src, left_bottom_src, right_top_src)
        
        left_bottom_tgt = sampled_feats_p_tgt[..., :, : self.output_dim]
        right_top_tgt = sampled_feats_p_tgt[..., :, self.output_dim:]
        emb_tgt = Hyper_Embedding.from_zZ(sampled_feats_I_tgt, left_bottom_tgt, right_top_tgt)

        return [emb_src, emb_tgt]
    
def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels=4160,
                                 out_channels=in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels=4160,
                                 out_channels=in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels=4160,
                                 out_channels=in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        out_channels=in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x1, x2):
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return h_