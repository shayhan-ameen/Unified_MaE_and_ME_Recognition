import torch
from torch import nn


class ST_Landmark_Att(nn.Module):
    """
    Spatio-Temporal Landmark Attention module.

    This module learns an attention map over both:

        1. Temporal dimension:
            Which keyframes are more important?

        2. Landmark dimension:
            Which facial landmarks are more important?

    It is inspired by coordinate attention. Instead of computing only channel
    attention, it separately summarizes the input feature map along the landmark
    and temporal dimensions, then combines both attention responses.

    Input shape:
        x: [B, C, T, V]

        B = batch size
        C = number of feature channels
        T = number of selected frames or keyframes
        V = number of facial landmarks, usually 51

    Output shape:
        out: [B, C, T, V]

    Main idea:
        x_t is obtained by average pooling over landmarks:

            x_t = mean(x, dim=V)

            Shape:
                [B, C, T, 1]

        x_v is obtained by average pooling over time:

            x_v = mean(x, dim=T)

            Shape before transpose:
                [B, C, 1, V]

            Shape after transpose:
                [B, C, V, 1]

        Then x_t and x_v are concatenated along the temporal-like dimension:

            [B, C, T + V, 1]

        The shared bottleneck layer reduces the channel dimension, and two
        separate 1 x 1 convolutions generate temporal and landmark attention.

    Attention generation:
        temporal attention:
            x_t_att: [B, C, T, 1]

        landmark attention:
            x_v_att: [B, C, 1, V]

        joint spatio-temporal attention:
            x_att = x_t_att * x_v_att

            Shape:
                [B, C, T, V]

    Final output:
        out = input * x_att

    Args:
        args:
            Configuration object. It should contain:

                args.reduct_ratio:
                    Channel reduction ratio for the bottleneck.

                args.bias:
                    Whether convolution layers use bias.

                args.act:
                    Activation function. Kept for compatibility.

        channel:
            Number of input channels C.
    """

    def __init__(self, args, channel):
        super(ST_Landmark_Att, self).__init__()

        # Bottleneck channel size.
        # max(1, ...) avoids zero channels when channel < reduct_ratio.
        inner_channel = max(1, channel // args.reduct_ratio)

        # Shared transformation layer.
        #
        # Input:
        #     [B, C, T + V, 1]
        #
        # Output:
        #     [B, inner_channel, T + V, 1]
        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=args.bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )

        # Temporal attention projection.
        #
        # Input:
        #     [B, inner_channel, T, 1]
        #
        # Output:
        #     [B, C, T, 1]
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1, bias=args.bias)

        # Landmark attention projection.
        #
        # Input:
        #     [B, inner_channel, 1, V]
        #
        # Output:
        #     [B, C, 1, V]
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1, bias=args.bias)

        # Kept for compatibility with your original implementation.
        # The fcn block already contains BatchNorm2d.
        self.bn = nn.BatchNorm2d(inner_channel)

        # Kept for compatibility, although Hardswish is used above.
        self.act = args.act

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input feature tensor with shape [B, C, T, V].

        Returns:
            out:
                Reweighted feature tensor with shape [B, C, T, V].
        """

        if x.dim() != 4:
            raise ValueError(
                f"Expected input shape [B, C, T, V], but got {tuple(x.shape)}."
            )

        # Save original features for final reweighting.
        res = x

        B, C, T, V = x.size()

        # ------------------------------------------------------------
        # 1. Temporal descriptor
        # ------------------------------------------------------------
        # Average over landmarks.
        #
        # Shape:
        #     [B, C, T, V] -> [B, C, T, 1]
        x_t = x.mean(dim=3, keepdim=True)

        # ------------------------------------------------------------
        # 2. Landmark descriptor
        # ------------------------------------------------------------
        # Average over time.
        #
        # Shape:
        #     [B, C, T, V] -> [B, C, 1, V]
        #
        # Then transpose to make it compatible with x_t for concatenation:
        #     [B, C, 1, V] -> [B, C, V, 1]
        x_v = x.mean(dim=2, keepdim=True).transpose(2, 3)

        # ------------------------------------------------------------
        # 3. Shared bottleneck transformation
        # ------------------------------------------------------------
        # Concatenate temporal and landmark descriptors.
        #
        # Shape:
        #     x_t: [B, C, T, 1]
        #     x_v: [B, C, V, 1]
        #
        #     concat: [B, C, T + V, 1]
        x_att = torch.cat([x_t, x_v], dim=2)

        # Reduce channel dimension and learn shared representation.
        #
        # Shape:
        #     [B, C, T + V, 1]
        #         -> [B, inner_channel, T + V, 1]
        x_att = self.fcn(x_att)

        # ------------------------------------------------------------
        # 4. Split temporal and landmark branches
        # ------------------------------------------------------------
        # x_t:
        #     [B, inner_channel, T, 1]
        #
        # x_v:
        #     [B, inner_channel, V, 1]
        x_t, x_v = torch.split(x_att, [T, V], dim=2)

        # ------------------------------------------------------------
        # 5. Generate temporal attention
        # ------------------------------------------------------------
        # Shape:
        #     [B, inner_channel, T, 1]
        #         -> [B, C, T, 1]
        x_t_att = self.conv_t(x_t).sigmoid()

        # ------------------------------------------------------------
        # 6. Generate landmark attention
        # ------------------------------------------------------------
        # First transpose:
        #     [B, inner_channel, V, 1]
        #         -> [B, inner_channel, 1, V]
        #
        # Then project:
        #     [B, inner_channel, 1, V]
        #         -> [B, C, 1, V]
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()

        # ------------------------------------------------------------
        # 7. Joint spatio-temporal attention
        # ------------------------------------------------------------
        # Broadcasting:
        #     x_t_att: [B, C, T, 1]
        #     x_v_att: [B, C, 1, V]
        #
        # Result:
        #     x_att:   [B, C, T, V]
        x_att = x_t_att * x_v_att

        # ------------------------------------------------------------
        # 8. Reweight input features
        # ------------------------------------------------------------
        # This matches the manuscript:
        #     output = input * attention
        out = res * x_att

        return out


class Channel_Att(nn.Module):
    """
    Channel attention module.

    This module learns channel-wise importance scores using global average
    pooling followed by a small bottleneck network.

    Input shape:
        x: [B, C, T, V]

    Output shape:
        attention: [B, C, 1, 1]

    Usage:
        If you want only the attention map:

            att = channel_att(x)

        If you want reweighted features:

            out = x * channel_att(x)

    Args:
        channel:
            Number of input channels C.
    """

    def __init__(self, channel, **kwargs):
        super(Channel_Att, self).__init__()

        inner_channel = max(1, channel // 4)

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inner_channel, kernel_size=1),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input tensor with shape [B, C, T, V].

        Returns:
            attention:
                Channel attention map with shape [B, C, 1, 1].
        """

        if x.dim() != 4:
            raise ValueError(
                f"Expected input shape [B, C, T, V], but got {tuple(x.shape)}."
            )

        return self.fcn(x)


class Frame_Att(nn.Module):
    """
    Frame attention module.

    This module learns temporal attention scores for frames or keyframes.

    It first treats the temporal dimension as the main dimension for attention,
    then applies average pooling and max pooling to produce two descriptors.
    These descriptors are concatenated and passed through a convolution layer.

    Input shape:
        x: [B, C, T, V]

    Output shape:
        attention: [B, 1, T, 1]

    Usage:
        If you want only the attention map:

            att = frame_att(x)

        If you want reweighted features:

            out = x * frame_att(x)

    Notes:
        The convolution kernel size is:

            kernel_size = (9, 1)

        This lets the module consider neighboring frames when estimating
        frame-level attention.
    """

    def __init__(self, **kwargs):
        super(Frame_Att, self).__init__()

        # Pool each temporal position into one value.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Temporal convolution over the frame dimension.
        self.conv = nn.Conv2d(2, 1, kernel_size=(9, 1), padding=(4, 0))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input tensor with shape [B, C, T, V].

        Returns:
            attention:
                Frame attention map with shape [B, 1, T, 1].
        """

        if x.dim() != 4:
            raise ValueError(
                f"Expected input shape [B, C, T, V], but got {tuple(x.shape)}."
            )

        # Original shape:
        #     [B, C, T, V]
        #
        # Transpose channel and temporal dimensions:
        #     [B, T, C, V]
        #
        # This makes each frame act like a channel for the pooling operation.
        x = x.transpose(1, 2)

        # Average and max descriptors.
        #
        # avg_pool(x): [B, T, 1, 1]
        # max_pool(x): [B, T, 1, 1]
        #
        # Concatenate along descriptor dimension:
        #     [B, T, 2, 1]
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=2)

        # Prepare for convolution:
        #     [B, T, 2, 1] -> [B, 2, T, 1]
        x = x.transpose(1, 2)

        # Output:
        #     [B, 1, T, 1]
        attention = self.conv(x)

        return attention
