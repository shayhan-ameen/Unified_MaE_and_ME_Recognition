import torch
from torch import nn

from .attentions import ST_Landmark_Att as Attention_Layer
from .layers import Spatial_Graph_Layer, Temporal_Basic_Layer, Temporal_Bottleneck_Layer, Temporal_Sep_Layer, Temporal_SG_Layer



class EfficientGCN(nn.Module):
    def __init__(self, args):
        super(EfficientGCN, self).__init__()
        # main stream
        self.main_stream = EfficientGCN_Blocks(args, block_args = args.model_block_args)
        # output
        last_channel = args.model_block_args[-1][0]
        self.classifier = EfficientGCN_Classifier(args, last_channel)
        # init parameters
        init_param(self.modules())

    def forward(self, x, Adj):
        x = self.main_stream(x, Adj)
        out = self.classifier(x)
        out = out.squeeze(dim=2)
        out = out.squeeze(dim=2)
        return out


class EfficientGCN_Blocks(nn.Module):
    # def __init__(self, init_channel, block_args, layer_type, kernel_size, input_channel=0, **kwargs):
    def __init__(self, args, block_args):
        super(EfficientGCN_Blocks, self).__init__()

        temporal_window_size, max_graph_distance = args.kernel_size

        last_channel = args.stream_input_channel
        temporal_layer = Temporal_Basic_Layer
        self.blocks = nn.ModuleList()

        for i, [output_channel, stride, depth] in enumerate(block_args):
            gcn = Spatial_Graph_Layer(args, last_channel, output_channel, max_graph_distance)
            tcn = nn.Sequential()
            for j in range(depth):
                s = stride if j == 0 else 1
                tcn.add_module(f'block-{i}_tcn-{j}', temporal_layer(args, output_channel, temporal_window_size, stride=s))
            attn = Attention_Layer( args, output_channel)
            last_channel = output_channel
            self.blocks.append(nn.ModuleDict({'gcn' : gcn, 'tcn' : tcn, 'attn' : attn}))


    def forward(self, x, Adj):
        for block in self.blocks:
            x = block['gcn'](x, Adj)
            x = block['tcn'](x)
            x = block['attn'](x)
        return x

class EfficientGCN_Classifier(nn.Sequential):
    def __init__(self, args, curr_channel):
        super(EfficientGCN_Classifier, self).__init__()

        num_class = 3 if args.num_classes == 'Folder' else int(args.num_classes)


        self.add_module('gap', nn.AdaptiveAvgPool2d(1))
        self.add_module('dropout', nn.Dropout(args.drop_prob, inplace=True))
        self.add_module('fc', nn.Conv2d(curr_channel, args.stream_embedding, kernel_size=1))

class EdgePredictor(nn.Module):
    def __init__(self, args):
        super(EdgePredictor, self).__init__()
        # main stream
        self.Encoder = EfficientGCN_Blocks(args, block_args = args.ep_block_args)
        self.B = nn.Parameter(torch.zeros((51,51)))
        # init parameters
        init_param(self.modules())

    def forward(self, x, Adj):
        x = self.Encoder(x, Adj)
        out = torch.einsum('nctv,gtcn->vg', (x, x.permute(*torch.arange(x.ndim - 1, -1, -1)))).contiguous()
        out = out @ self.B
        return out

def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
