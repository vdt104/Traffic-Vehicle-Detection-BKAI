
import torch
import torch.nn as nn

from ..modules.conv import Conv

# def autopad(k, p=None, d=1):  # kernel, padding, dilation
#     """Pad to 'same' shape outputs."""
#     if d > 1:
#         k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p

# class Conv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self.conv(x)))

#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(x))
    

class DWR(nn.Module):
    def __init__(self, c) -> None:
        super().__init__()
        self.conv_3x3 = Conv(c, c, 3, p=1)
        self.conv_3x3_d1 = Conv(c, c, 3, p=1, d=1)
        self.conv_3x3_d3 = Conv(c, c, 3, p=3, d=3)
        self.conv_3x3_d5 = Conv(c, c, 3, p=5, d=5)
        self.conv_1x1 = Conv(c * 3, c, 1)

    def forward(self, x):
        x_ = self.conv_3x3(x)
        x1 = self.conv_3x3_d1(x_)
        x2 = self.conv_3x3_d3(x_)
        x3 = self.conv_3x3_d5(x_)
        x_out = torch.cat([x1, x2, x3], dim=1)
        x_out = self.conv_1x1(x_out) + x
        return x_out
    
class DWRSeg_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1, dilation=1):
        super(DWRSeg_Conv, self).__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size)
        self.dcnv3 = DWR(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.dcnv3(x)
        x = self.gelu(self.bn(x))
        return x