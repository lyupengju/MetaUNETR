from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
# from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

rearrange, _ = optional_import("einops", name="rearrange")
from SwinUNETR import SwinTransformer
from Uxnet import  uxnet_conv
from  Sequencer import SequencerLSTM
from  Vip_3d import VisionPermutator
# from poolmixer import poolmixer
from GFnet import fft_mixer


''' downsampling methods'''
class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        # print(dim)
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):

        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x0 = x[:, 0::2, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, 0::2, :]
            x3 = x[:, 0::2, 0::2, 1::2, :]
            x4 = x[:, 1::2, 0::2, 1::2, :]
            x5 = x[:, 1::2, 1::2, 0::2, :]
            x6 = x[:, 0::2, 1::2, 1::2, :]
            x7 = x[:, 1::2, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x0 = x[:, 0::2, 0::2, :]
            x1 = x[:, 1::2, 0::2, :]
            x2 = x[:, 0::2, 1::2, :]
            x3 = x[:, 1::2, 1::2, :]
            x = torch.cat([x0, x1, x2, x3], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x



class downsample_conv(nn.Module):
    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        # print(dim)
        self.reduction = nn.Conv3d(dim,  2 * dim, kernel_size=2, stride=2) # 特征图图像减半
        self.norm = norm_layer(dim)
       
    def forward(self, x):
        # print('down') #b, d, h, w, c 
        x = self.norm(x)
        x = rearrange(x, "b d h w c -> b c d h w")

        # x = x.permute(0, 4, 1, 2, 3).contiguous()  #b, c, d, h, w 
        x = self.reduction(x)
        # x = x.permute(0,2,3,4,1).contiguous()  #b, d, h, w, c 
        x = rearrange(x, "b c d h w -> b d h w c")

        return x

MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2, "downsample_conv": downsample_conv}

class tokenmixers(nn.Module):

    def __init__(
        self,
        encoder: str,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24), #transformer
        hidden_sizes: Sequence[int]= [32, 64, 128, 256], #sequencer
        segment_dim: Sequence[int]=  [48, 24, 12, 6], #vip
        feature_size: int = 24, # base channel 
        mlp_ratio: int = 3,
        norm_name: Union[Tuple, str] = "instance",
        normalize: bool = True,  # norm encoder outputa
        spatial_dims: int = 3,
        downsample= "downsample_conv" #"merging",  下采样方法
    ) -> None:
       

        super().__init__()
       
        self.normalize = normalize #norm all encoder outputs
        if feature_size % 12 != 0:
                raise ValueError("feature_size should be divisible by 12.")
        
        if encoder == 'swintransformer':
            '''hyperparameters check'''
            img_size = ensure_tuple_rep(img_size, 3)
            patch_size = ensure_tuple_rep(2, 3)
            window_size = ensure_tuple_rep(7, 3)
            if not (spatial_dims == 2 or spatial_dims == 3):
                raise ValueError("spatial dimension should be 2 or 3.")
            for m, p in zip(img_size, patch_size):
                for i in range(5):
                    if m % np.power(p, i + 1) != 0:
                        raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")
            

            self.encoder_backbone = SwinTransformer(
                in_chans=in_channels,
                embed_dim=feature_size,
                window_size=window_size,
                patch_size=patch_size,
                depths=depths,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                patch_norm  = False,  # cancel patch norm
                spatial_dims=spatial_dims,
                downsample=look_up_option(downsample, MERGING_MODE)
            )
        elif encoder == 'large_kernel_Conv':
            self.encoder_backbone = uxnet_conv(in_channels,
                        downsample_method=look_up_option(downsample, MERGING_MODE),
                        depths=depths, embed_dim = feature_size, mlp_ratio = mlp_ratio)
 

        elif encoder == 'Sequencer':
            self.encoder_backbone = SequencerLSTM(
            in_chans=in_channels,
            depths=depths, # 4 层,每层 sequencer block数
            embed_dim=feature_size,
            hidden_sizes=hidden_sizes, #lstm 的hidden state size
            mlp_ratio= mlp_ratio,  
            union=  "cat", #"cat",
            num_rnn_layers=1, #RNN layer
            downsample_method=look_up_option(downsample, MERGING_MODE)

        )
            
        elif encoder == 'Vip':
           self.encoder_backbone =  VisionPermutator( in_chans=in_channels, depths=depths,  segment_dim = segment_dim , 
        embed_dim = feature_size,  mlp_ratio = mlp_ratio, downsample_method=look_up_option(downsample, MERGING_MODE))

        elif encoder == 'poolmixer':
           self.encoder_backbone = poolmixer(in_channels,
                        downsample_method=look_up_option(downsample, MERGING_MODE),
                        depths=depths, embed_dim = feature_size, mlp_ratio = mlp_ratio)

        elif encoder == 'fft_mixer':
            self.encoder_backbone =  fft_mixer( in_chans=in_channels, depths=depths,  size = [48, 24, 12, 6],
        embed_dim = feature_size,  mlp_ratio = mlp_ratio, downsample_method=look_up_option(downsample, MERGING_MODE))

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

   

    def forward(self, x_in):
        hidden_states_out = self.encoder_backbone(x_in, self.normalize)
        # for s in hidden_states_out:
        #     print(s.shape)
        # n, ch, d, h, w = x_in.size()
        # x_in = rearrange(x_in, "n c d h w -> n d h w c")
        # x_in = F.layer_norm(x_in, [ch])
        # x_in = rearrange(x_in, "n d h w c -> n c d h w")

        output = []

        enc0 = self.encoder1(x_in)
        output.append(enc0)
        enc1 = self.encoder2(hidden_states_out[0])
        output.append(enc1)
        enc2 = self.encoder3(hidden_states_out[1])
        output.append(enc2)
        enc3 = self.encoder4(hidden_states_out[2])
        output.append(enc3)
        enc4 = self.encoder5(hidden_states_out[3])
        output.append(enc4)
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits#, output


if __name__ == '__main__':   
    Unet = tokenmixers(
    encoder = 'fft_mixer',#'large_kernel_Conv', #'swintransformer',poolmixer, fft_mixer
    img_size=(96,96,96), #patch-size ,必须是32的倍数
    in_channels=1,
    out_channels=2,
    depths = (2, 2, 2, 2), # tokenmixer num per layer
    num_heads = (3, 6, 12, 24), # for transformer
    feature_size = 48,  #base channel
    hidden_sizes = [48, 96, 192, 384], #sequencer
    segment_dim = [48, 24, 12, 6], #vip
    mlp_ratio = 4,
)
      
    x = torch.ones((1,1,96,96,96))#注意h w 尺寸大小,magesize/ patchsize = segment_dimension,embed_dim 要能整除segment_dim
    y = Unet(x)
    
    print(y.shape)