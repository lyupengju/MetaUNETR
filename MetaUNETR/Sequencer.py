#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

import math
from functools import partial
from timm.layers import lecun_normal_, Mlp
from timm.models.helpers import build_model_with_cfg, named_apply
from torch import nn
import torch.nn.functional as F
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

from layers3D import Sequencer3DBlock, LSTM3D, GRU3D, RNN3D,VanillaSequencerBlock
rearrange, _ = optional_import("einops", name="rearrange")
__all__ = [
 
    "SequencerLSTM"
]

class BasicLayer(nn.Module):
    def __init__(self, dim, downsample, depth_per_layer = 2, mlp_ratio = 4.0,block_layer =  Sequencer3DBlock, union="cat", hidden_sizes= 20,
                rnn_layer = LSTM3D, num_rnn_layers = 1, drop_path_rate=0.,):
        super().__init__()

        self.layer = nn.Sequential(
                *[block_layer(dim, hidden_sizes, mlp_ratio = mlp_ratio,union =union,
                                  rnn_layer=rnn_layer,  num_layers=num_rnn_layers,
                                   drop_path=drop_path_rate) for j in range(depth_per_layer)])

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

class SequencerLSTM(nn.Module):
    def __init__(
            self,
            in_chans=1,
            depths=[2, 2, 2, 2], # 4 层,每层 sequencer block数
            embed_dim = 48,
            hidden_sizes= [48, 96, 192, 384], #lstm 的hidden state size
            union= "cat", #"add",
            mlp_ratio =[3.0, 3.0, 3.0, 3.0],  
            block_layer= Sequencer3DBlock,
            rnn_layer= LSTM3D, #GRU3D LSTM3D, VanillaSequencerBlock
            num_rnn_layers= 1, #RNN layer
            downsample_method = 'conv',

            
    ):
           
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
                hidden_sizes = hidden_sizes[i_layer],
                downsample = downsample_method,
                depth_per_layer = depths[i_layer],
                mlp_ratio = mlp_ratio,
                block_layer=block_layer,
                rnn_layer=rnn_layer,
                num_rnn_layers = num_rnn_layers,
                union = union
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
#     import torch
#     model = Sequencer(num_classes=14,
#             in_chans=1,
#             layers=[2, 2, 2, 2],
#             patch_sizes=[3, 2, 1, 1],
#             embed_dims=[192, 384, 384, 384],
#             hidden_sizes=[48, 96, 96, 96],
#             mlp_ratios=[3.0, 3.0, 3.0, 3.0],)
    
#     #注意h w 尺寸大小,imagesize/ patchsize = segment_dimension,embed_dim 要能整除segment_dim
#     # x = torch.ones((1,1,128,128,128))
#     x = torch.ones((1,1,96,96,96))

#     y = model(x)
   
