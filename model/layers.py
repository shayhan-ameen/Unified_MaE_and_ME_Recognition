import torch
from torch import nn
from torch.nn import functional as F


class Basic_Layer(nn.Module):
    def __init__(self, args, in_channel, out_channel, residual):
        super(Basic_Layer, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, bias=args.bias)
        self.out_channel = out_channel
        self.bn = nn.BatchNorm2d(out_channel)

        self.residual = nn.Identity() if residual else Zero_Layer()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x


class Spatial_Graph_Layer(nn.Module):
    def __init__(self, args, in_channel, out_channel, max_graph_distance, residual=True):
    # def __init__(self, in_channel, out_channel, max_graph_distance, bias, act, A, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__()

        # self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channel, out_channel * self.s_kernel_size, 1, bias=args.bias)

        self.residual = nn.Identity() if residual else Zero_Layer()
        if not residual:
            self.residual = Zero_Layer()
        elif residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=args.bias),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.residual = nn.Identity()

        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True)


    def forward(self, x, Adj):
        res = self.residual(x)
        x=self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,vv->nctv', (x, Adj)).contiguous()
        x = self.bn(x)
        x = x + res
        x = self.act(x)

        # x = self.act(self.bn(self.conv(x)) + res)
        return x

class Temporal_Basic_Layer(Basic_Layer):
    def __init__(self, args, channel, temporal_window_size, stride=1, residual=True):
        super(Temporal_Basic_Layer, self).__init__(args, channel, channel, residual)

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2d(channel, channel, (temporal_window_size,1), (stride,1), (padding,0), bias=args.bias)
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=args.bias),
                nn.BatchNorm2d(channel),
            )


class Temporal_Bottleneck_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Bottleneck_Layer, self).__init__()

        inner_channel = channel // reduct_ratio
        padding = (temporal_window_size - 1) // 2
        self.act = act

        self.reduct_conv = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.expand_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand_conv(x) + res)
        return x


class Temporal_Sep_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

        if expand_ratio > 0:
            inner_channel = channel * expand_ratio
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class Temporal_SG_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = channel // reduct_ratio
        self.act = act

        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size,1), 1, (padding,0), groups=channel, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size,1), (stride,1), (padding,0), groups=channel, bias=bias),
            nn.BatchNorm2d(channel),
        )

        if not residual:
            self.residual = Zero_Layer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res


class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0


# Thanks to YAN Sijie for the released code on Github (https://github.com/yysijie/st-gcn)
# class SpatialGraphConv(nn.Module):
#     def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, **kwargs):
#         super(SpatialGraphConv, self).__init__()
#
#         self.s_kernel_size = max_graph_distance + 1
#         self.gcn = nn.Conv2d(in_channel, out_channel*self.s_kernel_size, 1, bias=bias)
#         # self.A = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
#         self.A = A
#         if edge:
#             self.edge = nn.Parameter(torch.ones_like(self.A))
#         else:
#             self.edge = 1
#
#     def forward(self, x):
#
#         # print(f'1---------{x.shape=}')
#         x = self.gcn(x)
#         # print(f'2---------{x.shape=}')
#         n, kc, t, v = x.size()
#         x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
#         # print(f'3---------{x.shape=}')
#         # print(f'---------{self.A.shape=}')
#         # # print(f'---------{self.edge.shape=}')
#         # # x = torch.einsum('nkctv,kvw->nctw', (x, self.A * self.edge)).contiguous()
#         x = torch.einsum('nkctv,vv->nctv', (x, self.A * self.edge)).contiguous()
#         # x = torch.einsum('nkctv,ovv->nctv', (x, self.A * self.edge)).contiguous()
#         # x = torch.matmul(x, self.A)
#         # print(f'4---------{x.shape=}')
#         # print(f'---------{type(x)}')
#         return x


# class Spatial_Graph_Layer(Basic_Layer):
#     def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
#         super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)
#
#         self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
#         if residual and in_channel != out_channel:
#             self.residual = nn.Sequential(
#                 nn.Conv2d(in_channel, out_channel, 1, bias=bias),
#                 nn.BatchNorm2d(out_channel),
#             )



# class Spatial_Graph_Layer(nn.Module):
#     def __init__(self, in_channel, out_channel, max_graph_distance, bias, A, act, residual=True, **kwargs):
#         # def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, **kwargs):
#         # super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)
#         super(Spatial_Graph_Layer, self).__init__()
#
#         # self.conv = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
#         # self.out_channel = out_channel
#         self.s_kernel_size = max_graph_distance + 1
#         self.A = A
#
#         self.gcn = nn.Conv2d(in_channel, out_channel * self.s_kernel_size, 1, bias=bias)
#         self.bn = nn.BatchNorm2d(out_channel)
#
#         self.residual = nn.Identity() if residual else Zero_Layer()
#         if residual and in_channel != out_channel:
#             self.residual = nn.Sequential(
#                 nn.Conv2d(in_channel, out_channel, 1, bias=bias),
#                 nn.BatchNorm2d(out_channel),
#             )
#         self.act = act
#
#         def forward(self, x):
#             print('---------------1---------------')
#             res = self.residual(x)
#             x = self.gcn(x)
#             n, kc, t, v = x.size()
#             x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
#             # x = torch.einsum('nkctv,vv->nctv', (x, self.A)).contiguous()
#             x = x @ self.A
#             return x
#             # return self.act(self.bn(x) + res)