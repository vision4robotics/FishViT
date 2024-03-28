import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import einsum
from ..modules.conv import Conv, autopad
from ..modules.transformer import TransformerEncoderLayer
# from .attention import DAttention, FocusedLinearAttention, HiLo

__all__ = ['STattention']


######################################## STattention start ########################################

class STattention(nn.Module):
    """ Factorized attention with convolutional relative position encoding class. """

    def __init__(self, c1, num_heads=8, fc=256, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__()

        self.num_heads = num_heads
        self.c1 = c1
        head_dim = c1 // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(c1, c1 * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj3 = nn.Linear(c1, c1)
        self.proj_drop = nn.Dropout(proj_drop)

        # embedding
        self.proj1 = nn.Conv2d(c1, c1, kernel_size=1)
        self.norm = nn.LayerNorm(c1)

        # pooling positional encoding
        self.proj2 = nn.AdaptiveAvgPool2d((None, None))

        self.gap = nn.AdaptiveAvgPool2d((1, fc // num_heads))
        self.gap2 = nn.AdaptiveAvgPool2d((fc // num_heads, fc // num_heads))
        self.fc = nn.Sequential(
            nn.Linear(fc, fc),
            nn.BatchNorm1d(fc),
            nn.ReLU(inplace=True),
            nn.Linear(fc, fc),
            nn.Sigmoid()
        )


    def forward(self, x):

        # embedding
        _, _, H, W = x.shape
        x = self.proj1(x).flatten(2).transpose(1, 2)
        x = self.norm(x) # [16, 400, 256]

        # Pooling positional encoding.
        B, N, C = x.shape
        feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj2(feat) + feat
        x = x.flatten(2).transpose(1, 2)
        B, N, C = x.shape

        # Generate Q, K, V.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # Shape: [3, B, h, N, Ch].
        q, k, v = qkv[0], qkv[1], qkv[2]  # Shape: [B, h, N, Ch].

        # Factorized attention.
        k_softmax = k.softmax(dim=2)  # Softmax on dim N.
        k_softmax_T_dot_v = einsum('b h n k, b h n v -> b h k v', k_softmax, v)  # Shape: [B, h, Ch, Ch].
        factor_att = einsum('b h n k, b h k v -> b h n v', q, k_softmax_T_dot_v)  # Shape: [B, h, N, Ch].

        x = k_softmax_T_dot_v
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = x

        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.view(B, self.num_heads, C//self.num_heads)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = (torch.sign(factor_att) @ n_sub) + v

        # Output projection.
        x = self.scale * x
        x = x.transpose(1, 2).reshape(B, N, C)

        # Output projection.
        x = self.proj3(x)
        x = self.proj_drop(x)
        x = x.permute(0, 2, 1).view([-1, C, H, W]).contiguous()
        return x


######################################## STattention end ########################################
