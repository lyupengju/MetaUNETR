import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from monai.networks.blocks import MLPBlock as Mlp
from torch.cuda.amp import autocast
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from mamba_ssm import Mamba
rearrange, _ = optional_import("einops", name="rearrange")
__all__ = [
 
    "mamba_S4"
]

class Mamba3Dcross(nn.Module):

    def __init__(self, input_size: int, 
                 union="cat", with_fc=True,d_state = 16, d_conv = 4, expand = 2):
        super().__init__()

        self.input_size = input_size
        self.output_size = input_size
        self.union = union

        self.with_vertical = True
        self.with_horizontal = True
        self.with_depth = True

        self.with_fc = with_fc
        
        if with_fc:
            if union == "cat":
                self.fc = nn.Linear(3 * self.output_size, input_size)
            elif union == "add":
                self.fc = nn.Linear(self.output_size, input_size)
          
        elif union == "cat":
            pass
            if 2 * self.output_size != input_size:
                raise ValueError(f"The output channel {2 * self.output_size} is different from the input channel {input_size}.")
        elif union == "add":
            pass
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
        elif union == "vertical":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_horizontal = False
        elif union == "horizontal":
            if self.output_size != input_size:
                raise ValueError(f"The output channel {self.output_size} is different from the input channel {input_size}.")
            self.with_vertical = False
        else:
            raise ValueError("Unrecognized union: " + union)

        self.rnn_v = Mamba(
                d_model=   self.input_size, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.rnn_h = Mamba(
                d_model=   self.input_size, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.rnn_d = Mamba(
                d_model=   self.input_size, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )


    def forward(self, x):
        B, H, W, D, C = x.shape

        if self.with_vertical:
            v = x.permute(0, 2, 3, 1, 4).contiguous()
            v = v.reshape(-1, H, C)  #batch size, sequential length, input size
            v = self.rnn_v(v)
            v = v.reshape(B, W, D, H, -1)
            v = v.permute(0, 3 ,1, 2, 4).contiguous()
            # print('v.shape')
        if self.with_horizontal:
            h = x.permute(0, 1, 3, 2, 4).contiguous()
            h = h.reshape(-1, W, C)
            h = self.rnn_h(h) #output(batch_size, seq_len,  num_directions * hidden_size)
            # print(h.shape)
            h = h.reshape(B, H, D, W, -1)
            h = h.permute(0, 1 ,3, 2, 4).contiguous()

        if self.with_depth:
            d = x.reshape(-1, D, C)
            d = self.rnn_d(d)
            d = d.reshape(B, H, W, D, -1)
            # print('d.shape')

        if self.with_vertical and self.with_horizontal and self.with_depth:
            if self.union == "cat":
                x = torch.cat([v, h, d ], dim=-1)
                # print('v,h,d cat size')
                # print(x.shape)
            else:
                x = v + h + d
        elif self.with_vertical:
            x = v
        elif self.with_horizontal:
            x = h
        elif self.with_depth:
            x = d
        if self.with_fc:
            x = self.fc(x)
            # print('after one sequencer block')
            # print(x.shape)
        return x

# class MambaLayer(nn.Module):
#     def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
#         super().__init__()
       
#         self.mamba = Mamba(
#                 d_model=dim, # Model dimension d_model
#                 d_state=d_state,  # SSM state expansion factor
#                 d_conv=d_conv,    # Local convolution width
#                 expand=expand,    # Block expansion factor
#         )
    
#     @autocast(enabled=False)
#     def forward(self, x):
#         if x.dtype == torch.float16:
#             x = x.type(torch.float32)

#         x_mamba = self.mamba(x)

#         return x_mamba
    
class mamba_block(nn.Module):

    def __init__(self, dim,mlp_ratio,act = nn.GELU(), drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba3Dcross(input_size=dim)
        self.norm2 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_channels = Mlp(hidden_size=dim, mlp_dim=dim*mlp_ratio)
        
    def forward(self, x):
      
        x = x + self.drop_path(self.mamba(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    
        return x
    
class BasicLayer(nn.Module):
    def __init__(self, dim, downsample, depth_per_layer = 2, mlp_ratio = 4.0,
                 drop_path_rate=0.):
        super().__init__()

        self.layer = nn.Sequential(
                *[mamba_block(dim=dim, mlp_ratio = mlp_ratio, drop_path=drop_path_rate) for j in range(depth_per_layer)])

        if callable(downsample):
            self.downsample = downsample(dim=dim)
            
    def forward(self, x):
        x = rearrange(x, "b c d h w -> b d h w c")

        x= self.layer(x)
        x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")

        return x


class Mamba_cross(nn.Module):
  
   
    def __init__(self, in_chans, downsample_method, depths=[2,2,2,2], embed_dim = 48, mlp_ratio = 4.0):
        super().__init__()

        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=2, stride=2) 

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim = int(embed_dim * 2**i_layer),
                downsample = downsample_method,
                depth_per_layer = depths[i_layer],
                mlp_ratio = mlp_ratio
            )
                         
            if i_layer == 0:
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
    

