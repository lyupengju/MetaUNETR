import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath
from functools import partial
from monai.networks.blocks import MLPBlock as Mlp
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")
__all__ = [
 
    "uxnet_conv"
]

class conv_block(nn.Module):

    def __init__(self, dim,mlp_ratio,act = nn.GELU(), drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # self.dwconv = nn.Conv3d(dim, dim, kernel_size=13, padding=6, groups=dim) # groups=dim depthwise conv
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, stride = 1, padding=3, groups=dim) # groups=dim depthwise conv kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
        self.norm2 = nn.LayerNorm(dim)

       
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_channels = Mlp(hidden_size=dim, mlp_dim=dim*mlp_ratio)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None

        # self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        # self.act = nn.GELU()
        # # self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
    def forward(self, x):
        
        x = x + self.drop_path(self.dwconv(self.norm1(x).permute(0, 4, 1, 2, 3).contiguous()).permute(0, 2, 3, 4, 1).contiguous())
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
            # x (N, H, W, D, C)
        # if self.gamma is not None:  # layer_scale
        #     x = self.gamma * x
        # print(x.shape)

        # input = x     # (N, H, W, D, C)  b d h w c
        # x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 4, 1).contiguous() # (N, C, H, W, D) -> (N, H, W, D, C)
        # x = self.norm(x)
        # x = x .permute(0, 4, 1, 2, 3)
        # x = self.pwconv1(x)
        # x = self.act(x)
        # x = self.pwconv2(x)
        # x = x.permute(0, 2, 3, 4, 1)
        return x
    
class BasicLayer(nn.Module):
    def __init__(self, dim, downsample, depth_per_layer = 2, mlp_ratio = 4.0,
                 drop_path_rate=0.):
        super().__init__()

        self.layer = nn.Sequential(
                *[conv_block(dim=dim, mlp_ratio = mlp_ratio, drop_path=drop_path_rate) for j in range(depth_per_layer)])

        if callable(downsample):
            self.downsample = downsample(dim=dim)
            
    def forward(self, x):
        x = rearrange(x, "b c d h w -> b d h w c")

        x= self.layer(x)
        x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        # print('666')
        # print(x.shape)

        return x

        # b, d, h, w, c = x.size()




class uxnet_conv(nn.Module):
  
   
    def __init__(self, in_chans, downsample_method, depths=[2,2,2,2], embed_dim = 48, mlp_ratio = 4.0):
        super().__init__()

        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=2, stride=2) # 特征图图像减半， 等同于kernel =2 stride =2

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        for i_layer in range(len(depths)):
            # print(i_layer)
            layer = BasicLayer(
                dim = int(embed_dim * 2**i_layer),
                downsample = downsample_method,
                depth_per_layer = depths[i_layer],
                mlp_ratio = mlp_ratio
            )
                         
            if i_layer == 0:
                # print('555')
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            n, ch, d, h, w = x_shape
            x = rearrange(x, "n c d h w -> n d h w c")
            x = F.layer_norm(x, [ch])
            x = rearrange(x, "n d h w c -> n c d h w")

        return x

    def forward(self, x, normalize=True): #norm output
        # print(x.shape)
        x0 = self.patch_embed(x)
        x0_out = self.proj_out(x0, normalize) # patch norm
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize)
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x3 = self.layers3[0](x2.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]
    

