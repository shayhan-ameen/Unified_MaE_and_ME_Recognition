import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import ViTImageProcessor, ViTModel

# ============================================================
# Vision Transformer Stream
# ============================================================


class vit_model(nn.Module):
    """
    Vision Transformer feature extractor.

    This module uses a pretrained ViT model from HuggingFace:

        google/vit-large-patch16-224-in21k

    It extracts the [CLS] token representation from the ViT encoder and maps it
    to the desired final embedding size using a linear layer.

    Expected input:
        x: Tensor or image batch accepted by ViTImageProcessor.

        In this project, x usually has shape:

            [B, 3, H, W]

        where:
            B = batch size
            H = image height
            W = image width

    Output:
        logits: Tensor with shape [B, final_embedding_size]

    Notes:
        The input STLDN images are usually grayscale. In the visual stream,
        they are repeated across 3 channels before being passed to this module.
    """

    def __init__(self, final_embedding_size):
        super(vit_model, self).__init__()

        # HuggingFace image processor for ViT.
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-large-patch16-224-in21k"
        )

        # Pretrained ViT backbone.
        self.base_model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")

        # Hidden dimension of the ViT [CLS] token.
        self.hiddenDim = self.base_model.config.hidden_size

        # Projection layer to map ViT hidden dimension to stream embedding size.
        self.fc = nn.Linear(self.hiddenDim, final_embedding_size)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Image batch. Usually shape [B, 3, H, W].

        Returns:
            logits:
                Feature embedding with shape [B, final_embedding_size].
        """

        # Preprocess images for the ViT model.
        inputs = self.processor(
            images=x,
            return_tensors="pt",
            do_rescale=True,
        )

        # Move processor outputs to the same device as the model.
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward through pretrained ViT.
        outputs = self.base_model(**inputs)

        # Extract [CLS] token representation.
        cls_representation = outputs.last_hidden_state[:, 0, :]

        # Project [CLS] token to the desired embedding size.
        logits = self.fc(cls_representation)

        return logits


class lstm_model(nn.Module):
    """
    Multi-layer Bidirectional LSTM for temporal modeling.

    This module receives a sequence of frame-level features, usually extracted
    by the ViT model, and learns temporal dependencies across frames.

    Expected input:
        x: [B, S, D]

        B = batch size
        S = sequence length, usually num_frames - 1
        D = input feature size, default 128

    Output:
        x: [B, 128]

    Notes:
        Each LSTM layer is bidirectional. With hidden_size=64, the output
        feature dimension becomes:

            2 * hidden_size = 128

        After all LSTM layers, the sequence is flattened and projected to 128.
    """

    def __init__(self, num_layers=1, input_size=128, hidden_size=64, num_frames=10):
        super(lstm_model, self).__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
            )

        # Since BiLSTM output dimension is 2 * hidden_size = 128,
        # the flattened sequence size is:
        #
        #     (num_frames - 1) * 128
        self.lstm_fc = nn.Linear((num_frames - 1) * 128, 128)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input sequence tensor with shape [B, S, D].

        Returns:
            x:
                Temporal feature embedding with shape [B, 128].
        """

        for layer in self.layers:
            x, _ = layer(x)

        B, S, C = x.size()

        # Flatten temporal and channel dimensions.
        x = x.reshape(B, S * C)

        # Project to fixed 128-dimensional embedding.
        x = self.lstm_fc(x)

        return x


# ============================================================
# Attentive Feature Fusion
# ============================================================


class Multi_attentive_feature_fusion(nn.Module):
    """
    Attentive fusion module for three feature streams.

    This module fuses three feature vectors:

        1. s_f: first stream feature
        2. g_f: graph stream feature
        3. v_f: visual stream feature

    It learns one scalar attention weight per stream, then rescales each stream
    feature before concatenation and classification.

    Expected inputs:
        s_f: [B, in_channel]
        g_f: [B, in_channel]
        v_f: [B, in_channel]

    Output:
        x: [B, out_channel]

    Important:
        This module applies Softmax inside the classifier. If you use
        nn.CrossEntropyLoss, remove the final Softmax and return raw logits.
    """

    def __init__(self, in_channel=256, out_channel=2):
        super(Multi_attentive_feature_fusion, self).__init__()

        # Predict three attention scores, one for each stream.
        self.aff_model = nn.Sequential(
            nn.Linear(3 * in_channel, 3),
            nn.Softmax(dim=1),
        )

        # Final classifier after feature fusion.
        self.classifier = nn.Sequential(
            nn.Linear(3 * in_channel, out_channel),
            nn.Softmax(dim=1),
        )

    def forward(self, s_f, g_f, v_f):
        """
        Forward pass.

        Args:
            s_f:
                First stream feature with shape [B, in_channel].

            g_f:
                Graph stream feature with shape [B, in_channel].

            v_f:
                Visual stream feature with shape [B, in_channel].

        Returns:
            x:
                Class probability tensor with shape [B, out_channel].
        """

        # Concatenate features to estimate attention weights.
        x_att = torch.cat([s_f, g_f, v_f], dim=1)

        # Shape:
        #     [B, 3]
        x_att = self.aff_model(x_att)

        # Apply stream-specific attention weights.
        s_f = x_att[:, 0:1] * s_f
        g_f = x_att[:, 1:2] * g_f
        v_f = x_att[:, 2:3] * v_f

        # Concatenate reweighted features.
        x = torch.cat([s_f, g_f, v_f], dim=1)

        # Classify fused representation.
        x = self.classifier(x)

        return x


class attentive_feature_fusion(nn.Module):
    """
    Attentive Feature Fusion module for two streams.

    This module learns one scalar attention weight for each stream:
        1. Graph stream feature g_f
        2. Visual stream feature v_f

    Input:
        g_f: [B, C]
        v_f: [B, C]

    Output:
        x: [B, 2*C]
    """

    def __init__(self, in_channel=256, out_channel=2):
        super(attentive_feature_fusion, self).__init__()

        self.aff_model = nn.Sequential(nn.Linear(2 * in_channel, 2), nn.Softmax(dim=1))

    def forward(self, g_f, v_f):
        """
        Args:
            g_f:
                Graph stream feature with shape [B, C].

            v_f:
                Visual stream feature with shape [B, C].

        Returns:
            x:
                Fused feature with shape [B, 2*C].
        """

        # Estimate attention weights.
        # x_att shape: [B, 2]
        x_att = torch.cat([g_f, v_f], dim=1)
        x_att = self.aff_model(x_att)

        # Apply attention weights without creating diagonal matrices.
        # x_att[:, 0:1] shape: [B, 1], broadcast over feature dimension C.
        g_f = x_att[:, 0:1] * g_f
        v_f = x_att[:, 1:2] * v_f

        # Concatenate reweighted features.
        x = torch.cat([g_f, v_f], dim=1)

        return x


# ============================================================
# Spatio-Temporal Transformer
# ============================================================


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.

    This module supports both spatial attention and temporal attention by using
    flexible einops rearrangement patterns.

    Input:
        x: [B, sequence_length, input_dim]

    Output:
        x: [B, sequence_length, input_dim]

    Args:
        input_dim:
            Feature dimension of each token.

        dim_head:
            Feature dimension of each attention head.

        num_heads:
            Number of attention heads.

        drop_rate:
            Dropout probability.

    How it is used:
        In TransformerEncoder, this same module is used for:

            1. Spatial attention:
                Attention across landmarks within each frame.

            2. Temporal attention:
                Attention across frames for each landmark.
    """

    def __init__(
        self, input_dim: int, dim_head: int, num_heads: int, drop_rate: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.scale = dim_head ** (-0.5)

        # Linear projections for query, key, and value.
        self.q_w = nn.Linear(input_dim, dim_head * num_heads, bias=False)
        self.k_w = nn.Linear(input_dim, dim_head * num_heads, bias=False)
        self.v_w = nn.Linear(input_dim, dim_head * num_heads, bias=False)

        self.dropout = nn.Dropout(p=drop_rate)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)

        # Project concatenated heads back to input_dim.
        self.proj = nn.Linear(dim_head * num_heads, input_dim)

    def forward(self, x, einops_from, einops_to, **einops_dims):
        """
        Forward pass.

        Args:
            x:
                Input token sequence with shape [B, N, D].

            einops_from:
                Source einops pattern.

            einops_to:
                Target einops pattern.

            **einops_dims:
                Extra dimensions required by einops.

        Returns:
            x:
                Attention-enhanced tensor with shape [B, N, D].
        """

        residual = x

        # Project input into multi-head Q, K, V.
        q = rearrange(self.q_w(x), "b n (h d) -> (b h) n d", h=self.num_heads)
        k = rearrange(self.k_w(x), "b n (h d) -> (b h) n d", h=self.num_heads)
        v = rearrange(self.v_w(x), "b n (h d) -> (b h) n d", h=self.num_heads)

        # Rearrange tokens for spatial or temporal attention.
        q, k, v = map(
            lambda tensor: rearrange(
                tensor,
                f"{einops_from} -> {einops_to}",
                **einops_dims,
            ),
            (q, k, v),
        )

        # Scaled dot-product attention.
        attn = torch.einsum("bqd,bkd->bqk", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        # Weighted sum of values.
        x = torch.einsum("bqk,bkd->bqd", attn, v)

        # Restore previous token layout.
        x = rearrange(
            x,
            f"{einops_to} -> {einops_from}",
            **einops_dims,
        )

        # Merge attention heads.
        x = rearrange(
            x,
            "(b h) n d -> b n (h d)",
            h=self.num_heads,
        )

        # Output projection, dropout, residual connection, and normalization.
        x = self.dropout(self.proj(x)) + residual
        x = self.layer_norm(x)

        return x


class FeedForward(nn.Module):
    """
    Feed-forward network used inside the Transformer encoder.

    Structure:
        Linear -> ReLU -> Linear -> Dropout -> Residual -> LayerNorm

    Input:
        x: [B, N, input_dim]

    Output:
        x: [B, N, input_dim]
    """

    def __init__(self, input_dim: int, hidden_dim: int, drop_rate: float = 0.1):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(p=drop_rate),
        )

        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input tensor with shape [B, N, input_dim].

        Returns:
            x:
                Output tensor with shape [B, N, input_dim].
        """

        residual = x
        x = self.layer(x)
        x = x + residual
        x = self.layer_norm(x)

        return x


class TransformerEncoder(nn.Module):
    """
    Spatio-temporal Transformer encoder block.

    This block applies:

        1. Spatial attention over landmarks.
        2. Temporal attention over frames.
        3. Feed-forward network.

    Input:
        x: [B, C, T, V]

        B = batch size
        C = feature dimension
        T = number of frames
        V = number of landmarks

    Output:
        x: [B, C, T, V]
    """

    def __init__(
        self,
        input_dim: int,
        forward_dim: int,
        num_heads: int,
        head_dim: int,
        drop_rate: float = 0.1,
    ):
        super().__init__()

        self.sp_attn = MultiHeadAttention(
            input_dim=input_dim,
            dim_head=head_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
        )

        self.tm_attn = MultiHeadAttention(
            input_dim=input_dim,
            dim_head=head_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
        )

        self.feedforward = FeedForward(
            input_dim=input_dim,
            hidden_dim=forward_dim,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input tensor with shape [B, C, T, V].

        Returns:
            x:
                Output tensor with shape [B, C, T, V].
        """

        b, c, t, n = x.shape

        # Flatten temporal and landmark dimensions into tokens.
        x = rearrange(x, "b c t n -> b (t n) c").contiguous()

        # Spatial attention:
        # For each frame, attend across landmarks.
        x = self.sp_attn(
            x,
            "b (t n) d",
            "(b t) n d",
            t=t,
        )

        # Temporal attention:
        # For each landmark, attend across frames.
        x = self.tm_attn(
            x,
            "b (t n) d",
            "(b n) t d",
            n=n,
        )

        # Feed-forward network.
        x = self.feedforward(x)

        # Restore graph layout.
        x = rearrange(
            x,
            "b (t n) c -> b c t n",
            t=t,
            n=n,
        ).contiguous()

        return x


class TransformerClassifier(nn.Module):
    """
    Classifier head for spatio-temporal Transformer features.

    Input:
        x: [B, C, T, V]

    Processing:
        1. Mean pooling over T and V.
        2. LayerNorm.
        3. Linear projection.
        4. PReLU.
        5. Final linear classifier.

    Output:
        out: [B, num_classes]
    """

    def __init__(self, num_classes, input_dim=49, pool="mean", dropout=0.0):
        super().__init__()

        assert pool in {"cls", "mean"}, "pool type must be either cls or mean"

        self.to_latent = nn.Sequential(
            nn.Identity(),
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 2 * input_dim),
            nn.PReLU(),
        )

        self.mlp_head = nn.Linear(2 * input_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Transformer feature tensor with shape [B, C, T, V].

        Returns:
            out:
                Class logits with shape [B, num_classes].
        """

        # Global average pooling over temporal and landmark dimensions.
        cls_token = self.to_latent(x.mean(-1).mean(-1))

        out = self.mlp_head(cls_token)

        return out


class SpatioTemporalTransformer(nn.Module):
    """
    Full spatio-temporal Transformer model.

    This model stacks multiple TransformerEncoder blocks and uses a classifier
    head to predict expression classes.

    Input:
        x: [B, C, T, V]

    Output:
        x: [B, num_classes]
    """

    def __init__(
        self,
        num_layers,
        input_dim,
        forward_dim,
        num_heads=8,
        head_dim=16,
        drop_rate=0.1,
        num_classes=3,
    ):
        super().__init__()

        self.TransformerEncoder = nn.Sequential(
            *[
                TransformerEncoder(
                    input_dim,
                    forward_dim,
                    num_heads,
                    head_dim,
                    drop_rate,
                )
                for _ in range(num_layers)
            ]
        )

        self.TransformerClassifier = TransformerClassifier(
            num_classes,
            input_dim,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input tensor with shape [B, C, T, V].

        Returns:
            x:
                Class logits with shape [B, num_classes].
        """

        x = self.TransformerEncoder(x)
        x = self.TransformerClassifier(x)

        return x


# ============================================================
# 3D ResNet-101
# ============================================================


def conv3x3x3(in_planes, out_planes, stride=1):
    """
    Create a 3 x 3 x 3 convolution layer with padding.

    Args:
        in_planes:
            Number of input channels.

        out_planes:
            Number of output channels.

        stride:
            Convolution stride.

    Returns:
        nn.Conv3d layer.
    """

    return nn.Conv3d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class Bottleneck(nn.Module):
    """
    Bottleneck block for 3D ResNet.

    Structure:
        1 x 1 x 1 conv
        3 x 3 x 3 conv
        1 x 1 x 1 conv
        residual connection
        ReLU

    Input:
        x: [B, C, D, H, W]

    Output:
        out: [B, planes * expansion, D_out, H_out, W_out]

    where:
        D = temporal depth or number of frames
    """

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.stride = stride

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # Downsample is used when residual and main branch shapes differ.
        self.downsample = downsample

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input tensor with shape [B, C, D, H, W].

        Returns:
            out:
                Output tensor after bottleneck residual block.
        """

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

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    3D ResNet model.

    Input format:
        x: [B, C, D, H, W]

        B = batch size
        C = input channels
        D = temporal depth, number of frames
        H = frame height
        W = frame width

    Output:
        x: [B, num_classes]

    Args:
        layers:
            Number of bottleneck blocks in each ResNet stage.

        sample_size:
            Input spatial size.

        sample_duration:
            Input temporal duration.

        block:
            Residual block type. Default is Bottleneck.

        input_dim:
            Number of input channels.

        shortcut_type:
            Shortcut type. Type B uses projection shortcut.

        num_classes:
            Number of output classes.
    """

    def __init__(
        self,
        layers,
        sample_size,
        sample_duration,
        block=Bottleneck,
        input_dim=1,
        shortcut_type="B",
        num_classes=8,
    ):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv3d(
            input_dim,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(
            kernel_size=(3, 3, 3),
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))

        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size),
            stride=1,
        )

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Parameter initialization.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        """
        Build one ResNet stage.

        Args:
            block:
                Residual block class.

            planes:
                Base channel size for the stage.

            blocks:
                Number of blocks in the stage.

            shortcut_type:
                Shortcut type.

            stride:
                Stride for the first block.

        Returns:
            nn.Sequential containing the stage blocks.
        """

        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                assert True, "Shortcut type A is not implemented."
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []

        layers.append(
            block(
                self.in_planes,
                planes,
                stride,
                downsample,
            )
        )

        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input video tensor with shape [B, C, D, H, W].

        Returns:
            x:
                Class logits with shape [B, num_classes].
        """

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
    """
    Construct a 3D ResNet-101 model.

    Args:
        n_classes:
            Number of expression classes.

        sample_duration:
            Number of input frames.

        sample_size:
            Input frame size.

    Returns:
        model:
            3D ResNet-101 model.
    """

    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        shortcut_type="B",
        num_classes=n_classes,
        sample_duration=sample_duration,
        sample_size=sample_size,
    )

    return model


# ============================================================
# Simple GCN Baseline
# ============================================================


class GCNLayer(nn.Module):
    """
    Single graph convolution layer.

    Operation:
        X' = activation(A X W + b)

    Input:
        h: [B, V, input_dim]
        adj: [V, V] or [B, V, V]

    Output:
        x: [B, V, output_dim]

    Args:
        input_dim:
            Input node feature dimension.

        output_dim:
            Output node feature dimension.

        activation:
            Activation function, for example F.elu.

        dropout:
            Dropout probability or False.

        bias:
            Whether to use learnable bias.
    """

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
        """
        Initialize parameters.

        Weight matrix:
            Xavier uniform initialization.

        Bias:
            Initialized to zero.
        """

        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, h, adj):
        """
        Forward pass.

        Args:
            h:
                Node feature tensor with shape [B, V, input_dim].

            adj:
                Adjacency matrix with shape [V, V] or [B, V, V].

        Returns:
            x:
                Updated node feature tensor with shape [B, V, output_dim].
        """

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
    """
    Simple two-layer GCN classifier.

    This is a baseline model that applies two GCN layers and then flattens all
    node features for classification.

    Expected input:
        x: [B, V, input_dim]

        V is expected to be 51 in the current implementation.

    Adj:
        Adjacency matrix with shape [V, V] or [B, V, V]

    Output:
        x: [B, 3]

    Important:
        The final Softmax is included. If you train with nn.CrossEntropyLoss,
        remove Softmax and return raw logits.
    """

    def __init__(
        self,
        input_dim,
        intermediate_dim,
        out_dim,
        activation=F.elu,
        dropout=False,
        bias=True,
    ):
        super(SPECIAL_GCN, self).__init__()

        self.GCNLayer1 = GCNLayer(
            input_dim,
            intermediate_dim,
            activation,
            dropout,
            bias,
        )

        self.GCNLayer2 = GCNLayer(
            intermediate_dim,
            out_dim,
            activation,
            dropout,
            bias,
        )

        self.Linear = nn.Linear(51 * out_dim, 3)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x, Adj):
        """
        Forward pass.

        Args:
            x:
                Input node features with shape [B, 51, input_dim].

            Adj:
                Adjacency matrix with shape [51, 51] or [B, 51, 51].

        Returns:
            x:
                Class probabilities with shape [B, 3].
        """

        x = self.GCNLayer1(x, Adj)
        x = self.GCNLayer2(x, Adj)

        # Flatten all landmark node features.
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.Linear(x)
        x = self.Softmax(x)

        return x


class coordinates_model(nn.Module):
    """
    Coordinate feature projection model.

    This module maps 2D landmark coordinates to a higher-dimensional feature
    representation.

    Expected input:
        x: [B, V, 2]

        V is expected to be 153 in the current BatchNorm setting.

    Output:
        x: [B, V, out_channels]

    Args:
        in_channels:
            Input coordinate dimension. Default is 2 for x and y coordinates.

        out_channels:
            Output feature dimension for each coordinate. Default is 49.
    """

    def __init__(self, in_channels=2, out_channels=49):
        super(coordinates_model, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(153),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Coordinate tensor with shape [B, V, 2].

        Returns:
            x:
                Projected coordinate feature tensor.
        """

        return self.block(x)
