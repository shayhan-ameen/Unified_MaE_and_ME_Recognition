import torch
from torch import nn

class ST_Landmark_Att(nn.Module):
    """
    Spatial-Temporal Landmark Attention Module

    This module applies spatial-temporal attention to landmark features in a neural network. 
    The attention mechanism emphasizes important temporal and spatial features by generating 
    attention maps for both the temporal and spatial dimensions.

    Parameters:
    - args: A namespace containing various configuration arguments, including the reduction ratio, bias, and activation function.
    - channel (int): The number of input channels/features in the input tensor.

    Attributes:
    - fcn (nn.Sequential): A sequential module consisting of a convolutional layer, batch normalization, and an activation function, used to reduce the channel dimension.
    - conv_t (nn.Conv2d): A convolutional layer applied to the temporal attention map.
    - conv_v (nn.Conv2d): A convolutional layer applied to the spatial (landmark) attention map.
    - bn (nn.BatchNorm2d): A batch normalization layer applied after reducing the channel dimension.
    - act: The activation function specified in `args`.

    """

    def __init__(self, args, channel):
        super(ST_Landmark_Att, self).__init__()

        inner_channel = channel // args.reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=args.bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.bn = nn.BatchNorm2d(inner_channel)
        self.act = args.act

    def forward(self, x):
        res = x

        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att

        return x_att

class Channel_Att(nn.Module):
    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel//4, kernel_size=1),
            nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//4, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fcn(x)

class Frame_Att(nn.Module):
    def __init__(self, **kwargs):
        super(Frame_Att, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=(9,1), padding=(4,0))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=2).transpose(1, 2)
        return self.conv(x)
