import torch
import torch.nn as nn
from einops import rearrange
from transformers import ViTImageProcessor, ViTModel


class vit_model(nn.Module):
    def __init__(self, final_embedding_size):
        super(vit_model, self).__init__()

        # Initialize the image processor and base ViT model
        self.processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')
        self.base_model = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
        self.hiddenDim = self.base_model.config.hidden_size

        # Fully connected layer for classification
        self.fc = nn.Linear(self.hiddenDim, final_embedding_size)  # Use hiddenDim for the [CLS] token

    def forward(self, x):
        # Preprocess images using the processor
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=True)  # Consider rescaling

        # Send inputs to the same device as the model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Pass the inputs through the base ViT model
        outputs = self.base_model(**inputs)

        # Use the [CLS] token's representation for classification
        cls_representation = outputs.last_hidden_state[:, 0, :]  # Extract the [CLS] token representation

        # Pass the [CLS] token's representation through the fully connected layer
        logits = self.fc(cls_representation)

        return logits



class lstm_model(nn.Module):
    def __init__(self, num_layers=1, input_size=128, hidden_size=64, num_frames=10):
        super(lstm_model, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True))
        self.lstm_fc = nn.Linear((num_frames - 1) * 128, 128)
        # self.classification = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.layers:
            # print(f'{x.shape=}')
            x, _ = layer(x)
        N, S, C = x.size()
        x = x.reshape(N, S * C)
        x = self.lstm_fc(x)
        # x = self.classification(x)
        return x


class Multi_attentive_feature_fusion(nn.Module):
    def __init__(self, in_channel=256, out_channel=2):
        super(Multi_attentive_feature_fusion, self).__init__()
        self.aff_model = nn.Sequential(
            nn.Linear(3*in_channel, 3),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(3*in_channel, out_channel),
            nn.Softmax(dim=1)
        )

    def forward(self, s_f, g_f, v_f):
        x_att = torch.cat([s_f, g_f, v_f], dim=1)
        x_att = self.aff_model(x_att)
        att = torch.diag(x_att[:, 0])
        s_f = att @ s_f
        att = torch.diag(x_att[:, 1])
        g_f = att @ g_f
        att = torch.diag(x_att[:, 2])
        v_f = att @ v_f
        x = torch.cat([s_f, g_f, v_f], dim=1)
        x = self.classifier(x)
        return x

class attentive_feature_fusion(nn.Module):
    def __init__(self, in_channel=256, out_channel=2):
        super(attentive_feature_fusion, self).__init__()
        self.aff_model = nn.Sequential(
            nn.Linear(2*in_channel, 2),
            nn.Softmax(dim=1)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(2*in_channel, out_channel),
        #     nn.Softmax(dim=1)
        # )

    def forward(self, g_f, v_f):
        x_att = torch.cat([g_f, v_f], dim=1)
        x_att = self.aff_model(x_att)
        att = torch.diag(x_att[:, 0])
        g_f = att @ g_f
        att = torch.diag(x_att[:, 1])
        v_f = att @ v_f
        x = torch.cat([g_f, v_f], dim=1)
        # x = self.classifier(x)
        return x

# Spatio_Temporal_Transformer

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, dim_head: int, num_heads: int, drop_rate: float = 0.1):
        super().__init__()

        self.num_heads = num_heads #8
        self.scale = dim_head ** (-0.5) #0.25

        self.q_w = nn.Linear(in_features=input_dim, out_features=dim_head * num_heads, bias=False) # in_features=49, out_features=head_dim * num_heads = 128
        self.k_w = nn.Linear(in_features=input_dim, out_features=dim_head * num_heads, bias=False)
        self.v_w = nn.Linear(in_features=input_dim, out_features=dim_head * num_heads, bias=False)

        self.dropout = nn.Dropout(p=drop_rate)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)

        self.proj = nn.Linear(in_features=dim_head * num_heads, out_features=input_dim)

    def forward(self, x, einops_from, einops_to, **einops_dims):
        # Shape of x: (batch_size, num_frames * num_landmarks, input_dim) | 'b c t n -> b (t n) c'
        # Shape of residual: (batch_size, seq_length, input_dim) # size of x [1, 30, 49]

        residual = x

        # Make the dim head
        # Shape of q: (batch_size, num_heads, q_seq_length, head_dim)
        # Shape of k: (batch_size, num_heads, k_seq_length, head_dim)
        # Shape of v: (batch_size, num_heads, v_seq_length, head_dim)
        # NOTE: k_seq_length == v_seq_length
        # q = einops.rearrange(self.q_w(x), "b s (n d) -> b n s d", n=self.num_heads)  # [1, 30, 49] -> [1, 30, 128]) -> [1, 8, 30, 16]

        q = rearrange(self.q_w(x), 'b n (h d) -> (b h) n d', h=self.num_heads)  # [1, 30, 49] -> [1, 30, 128]) -> [8, 30, 16]
        k = rearrange(self.k_w(x), 'b n (h d) -> (b h) n d', h=self.num_heads)
        v = rearrange(self.v_w(x), 'b n (h d) -> (b h) n d', h=self.num_heads)

        # rearrange across time or space
        q, k, v = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))

        # Compute the attention energy
        # Shape of attn: (batch_size, num_heads, q_seq_length, k_seq_length)
        attn = torch.einsum("bqd,bkd->bqk", q, k) * self.scale # ([8, 30, 16], [8, 30, 16])-> [8, 30, 30]
        attn = attn.softmax(dim=-1)

        # Compute the final weight on value
        # Shape of x: (batch_size, q_seq_length, head_dim * num_heads)
        x = torch.einsum("bqk,bkd->bqd", attn, v) # ([1, 8, 30, 30], [1, 8, 30, 16])-> [1, 8, 30, 16]

        # merge back time or space
        x= rearrange(x, f'{einops_to} -> {einops_from}', **einops_dims)

        # merge back the heads
        x = rearrange(x, '(b h) n d -> b n (h d)', h=self.num_heads) # -> [1, 30, 128]

        # Shape of x: (batch_size, q_seq_length, input_dim)
        x = self.dropout(self.proj(x)) + residual # proj [1, 30, 128] ->[1, 30, 49]
        x = self.layer_norm(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 drop_rate: float = 0.1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim, out_features=input_dim),
            nn.Dropout(p=drop_rate)
        )
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        # Shape of residual: (batch_size, input_dim)
        residual = x

        # Before: Shape of x: (batch_size, input_dim)
        # After: Shape of x: (batch_size, input_dim)
        x = self.layer(x)
        x += residual

        x = self.layer_norm(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, forward_dim: int, num_heads: int, head_dim: int, drop_rate: float = 0.1):
        super().__init__()

        self.sp_attn = MultiHeadAttention(input_dim=input_dim, dim_head=head_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.tm_attn = MultiHeadAttention(input_dim=input_dim, dim_head=head_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.feedforward = FeedForward(input_dim=input_dim, hidden_dim=forward_dim, drop_rate=drop_rate)

    def forward(self, x):
        b, c, t, n = x.shape
        x = rearrange(x, 'b c t n -> b (t n) c').contiguous()
        x = self.sp_attn(x, 'b (t n) d', '(b t) n d', t=t)
        x = self.tm_attn(x, 'b (t n) d', '(b n) t d', n=n)
        x = self.feedforward(x)
        # To Vertexes
        x = rearrange(x, 'b (t n) c -> b c t n', t=t, n=n).contiguous()  # b c t n
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=49, pool='mean', dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_latent = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 2 * input_dim),
            nn.PReLU(),
        )
        self.mlp_head = nn.Linear(2 * input_dim, num_classes)

    def forward(self, x):
        cls_token = self.to_latent(x.mean(-1).mean(-1))  # b c
        out = self.mlp_head(cls_token)
        return out
    # return cls_token, out

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_layers, input_dim, forward_dim, num_heads=8, head_dim=16, drop_rate=0.1, num_classes=3):
        super().__init__()

        self.TransformerEncoder = nn.Sequential(*[TransformerEncoder(input_dim, forward_dim, num_heads, head_dim, drop_rate) for _ in range(num_layers)])
        self.TransformerClassifier = TransformerClassifier(num_classes, input_dim)

    def forward(self, x):
        x = self.TransformerEncoder(x)
        x = self.TransformerClassifier(x)
        return x



# b, n, c, h, w, t = vis.size()
#  b, n, _, h = *x.shape, self.head
# x_token = rearrange(x_token, 'b c t n -> b (t n) c').contiguous()
# x_token = rearrange(x_token, 'b (t n) c -> b c t n', t=t, n=n).contiguous()  # b c t n

# 3D- Resnet 101

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    # input format N,C,D,H,W = Batch size, Number of channels, Number of frames, Height, Width
    def __init__(self,
                 layers,
                 sample_size,
                 sample_duration,
                 block=Bottleneck,
                 input_dim=1,
                 shortcut_type='B',
                 num_classes=8):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.conv1 = nn.Conv3d(input_dim, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        # downsample case
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                assert True, 'Not implemented!'
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet101(n_classes, sample_duration, sample_size):
    """Constructs a 3D ResNet-101 model."""
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3], shortcut_type='B', num_classes=n_classes,
                   sample_duration=sample_duration, sample_size=sample_size)
    return model

# model = ResNet(input_dim=1,block=Bottleneck, layers=[3, 4, 23, 3], shortcut_type='B', num_classes=3, sample_duration=3, sample_size=225)


#--------- GCN model

import torch.nn.functional as F

class GCNLayer(nn.Module):
    """ one layer of GCN """
    def __init__(self, input_dim, output_dim, activation, dropout=False, bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.activation = activation
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.b = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, h, adj):
        if self.dropout:
            h = self.dropout(h)
        x = h @ self.W
        x = adj @ x
        if self.b is not None:
            x = x + self.b
        if self.activation:
            x = self.activation(x)
        return x


class SPECIAL_GCN(nn.Module):
    def __init__(self, input_dim, intermediate_dim, out_dim, activation=F.elu, dropout=False, bias=True):
        super(SPECIAL_GCN, self).__init__()

        self.GCNLayer1 = GCNLayer(input_dim, intermediate_dim, activation, dropout, bias)
        self.GCNLayer2 = GCNLayer(intermediate_dim, out_dim, activation, dropout, bias)
        self.Linear = nn.Linear(51*out_dim, 3)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x, Adj):
        x = self.GCNLayer1(x, Adj)
        x = self.GCNLayer2(x, Adj)
        # print(f'0--{x.shape=}')
        x = torch.flatten(x, start_dim=1, end_dim=- 1)
        # print(f'1--{x.shape=}')
        x = self.Linear(x)
        # print(f'2--{x.shape=}')
        x = self.Softmax(x)
        # print(f'3--{x.shape=}')
        return x

class coordinates_model(nn.Module):
    def __init__(self, in_channels=2, out_channels=49):
        super(coordinates_model, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(153),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
