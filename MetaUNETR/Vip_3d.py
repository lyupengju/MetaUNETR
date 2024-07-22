import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")
__all__ = [
 
    "VisionPermutator"
]

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, segment_dim=8,  proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.mlp_c = nn.Linear(dim, dim)
        self.mlp_h = nn.Linear(dim, dim)
        self.mlp_w = nn.Linear(dim, dim)
        self.mlp_d = nn.Linear(dim, dim)


        self.reweight = Mlp(dim, dim // 4, dim *4)
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x):
        B, H, W, D, C = x.shape

        S = C // self.segment_dim
        h = x.reshape(B, H, W, D, self.segment_dim, S).permute(0, 4, 2, 3, 1, 5).reshape(B, self.segment_dim, W, D, H*S)
        h = self.mlp_h(h).reshape(B, self.segment_dim, W, D, H, S).permute(0, 4, 2, 3, 1, 5).reshape(B, H, W, D, C)

        w = x.reshape(B, H, W, D, self.segment_dim, S).permute(0, 1, 4, 3, 2, 5).reshape(B, H, self.segment_dim, D, W*S)
        w = self.mlp_w(w).reshape(B, H, self.segment_dim, D, W, S).permute(0, 1, 4, 3, 2, 5).reshape(B, H, W, D, C)

        d = x.reshape(B, H, W, D, self.segment_dim, S).permute(0, 1, 2, 4, 3, 5).reshape(B, H, W, self.segment_dim, D*S)
        d = self.mlp_d(d).reshape(B, H, W, self.segment_dim, D, S).permute(0, 1, 2, 4, 3, 5).reshape(B, H, W, D, C)

        c = self.mlp_c(x)
        
        a = (h + w + d + c).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 4).permute(2, 0, 1).softmax(dim=0)
        a = a.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        x = h * a[0] + w * a[1] + d * a[2] + c * a[3]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4, 
                 drop_path=0.,  norm_layer=nn.LayerNorm, mlp_fn = WeightedPermuteMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.vip = mlp_fn(dim, segment_dim=segment_dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp( dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.vip(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x
    
class BasicLayer(nn.Module):
    def __init__(self, dim, downsample,segment_dim, depth_per_layer = 2, mlp_ratio = 4.0,
               mlp_fn = WeightedPermuteMLP):
        super().__init__()

        self.layer = nn.Sequential(
                *[PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio,
             mlp_fn = mlp_fn) for j in range(depth_per_layer)])

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

class VisionPermutator(nn.Module):

    def __init__(self, in_chans=3, depths=[2, 2, 2, 2],  segment_dim = [48, 24, 12, 6], 
        embed_dim = None,  mlp_ratio = 4, downsample_method = 'conv',):
  
        super().__init__()

        self.patch_embed = nn.Conv3d(in_chans, embed_dim, kernel_size=2, stride=2) # 特征图图像减半， 等同于kernel =2 stride =2
        # network.append(self.stem)   
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
                mlp_ratio = mlp_ratio,
                segment_dim =  segment_dim[i_layer],
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






# if __name__ == '__main__':   
    # model = vip3d(pretrained=False,base_embed_dims=48)

    # x = torch.ones((1,1,96,96,96))#注意h w 尺寸大小,magesize/ patchsize = segment_dimension,embed_dim 要能整除segment_dim
    # y = model(x)

    # print(y.shape)
