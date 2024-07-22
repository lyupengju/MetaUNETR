#  Copyright (c) 2022. Yuki Tatsunami
#  Licensed under the Apache License, Version 2.0 (the "License");

from functools import partial
from typing import Tuple
import torch
from timm.models.layers import DropPath, Mlp
from torch import nn, _assert, Tensor

class RNNIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(RNNIdentity, self).__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, None]:
        return x, None

class RNN3DBase(nn.Module):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2*hidden_size if bidirectional else hidden_size
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

        self.rnn_v = RNNIdentity()
        self.rnn_h = RNNIdentity()
        self.rnn_d = RNNIdentity()


    def forward(self, x):
        B, H, W, D, C = x.shape

        if self.with_vertical:
            v = x.permute(0, 2, 3, 1, 4).contiguous()
            v = v.reshape(-1, H, C)  #batch size, sequential length, input size
            v, _ = self.rnn_v(v)
            v = v.reshape(B, W, D, H, -1)
            v = v.permute(0, 3 ,1, 2, 4).contiguous()
            # print('v.shape')
        if self.with_horizontal:
            h = x.permute(0, 1, 3, 2, 4).contiguous()
            h = h.reshape(-1, W, C)
            h, _ = self.rnn_h(h) #output(batch_size, seq_len,  num_directions * hidden_size)
            # print(h.shape)
            h = h.reshape(B, H, D, W, -1)
            h = h.permute(0, 1 ,3, 2, 4).contiguous()

        if self.with_depth:
            d = x.reshape(-1, D, C)
            d, _ = self.rnn_d(d)
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


class RNN3D(RNN3DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)
        if self.with_horizontal:
            self.rnn_h = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)
        if self.with_depth:
            self.rnn_d = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional, nonlinearity=nonlinearity)



class LSTM3D(RNN3DBase): #因为是bidirection， 输出channel数为hidden size的 2倍

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_depth:
            self.rnn_d = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class GRU3D(RNN3DBase):

    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, bias: bool = True, bidirectional: bool = True,
                 union="cat", with_fc=True):
        super().__init__(input_size, hidden_size, num_layers, bias, bidirectional, union, with_fc)
        if self.with_vertical:
            self.rnn_v = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_horizontal:
            self.rnn_h = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)
        if self.with_depth:
            self.rnn_d = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bias=bias, bidirectional=bidirectional)


class VanillaSequencerBlock(nn.Module):
    def __init__(self, dim, hidden_size, mlp_ratio=3.0, rnn_layer=LSTM3D, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, drop=0., drop_path=0.,union="cat",with_fc= True):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Sequencer3DBlock(nn.Module):
    def __init__(self, dim, hidden_size, mlp_ratio=3.0, rnn_layer=LSTM3D, mlp_layer=Mlp,
                 norm_layer=partial(nn.LayerNorm), act_layer=nn.GELU,
                 num_layers=1, bidirectional=True, union="cat", with_fc=True,
                 drop=0., drop_path=0.):
        super().__init__()
        channels_dim = int(mlp_ratio * dim)
        self.norm1 = norm_layer(dim)
        self.rnn_tokens = rnn_layer(dim, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                                    union=union, with_fc=with_fc)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop) #output channel = input channel

    def forward(self, x):
        x = x + self.drop_path(self.rnn_tokens(self.norm1(x)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    
        return x




# class Downsample3D(nn.Module):
#     def __init__(self, input_dim, output_dim, patch_size):
#         super().__init__()
#         self.down = nn.Conv3d(input_dim, output_dim, kernel_size=patch_size, stride=patch_size)
#         # self.norm = nn.LayerNorm(output_dim)

#     def forward(self, x):
#         x = x.permute(0, 4, 1, 2, 3).contiguous()
#         x = self.down(x)
#         x = x.permute(0, 2, 3, 4, 1).contiguous()
#         # x = self.norm(x)
#         # print('after downsampling')
#         # print(x.shape)
#         return x



