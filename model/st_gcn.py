import math

import torch
from torch import nn

from .attentions import ST_Landmark_Att as Attention_Layer
from .layers import (
    Spatial_Graph_Layer,
    Temporal_Basic_Layer,
)


class EfficientGCN(nn.Module):
    """
    EfficientGCN main network.

    This module is used as the graph-based stream for expression recognition.
    It receives landmark-level node features and an adjacency matrix, then
    extracts spatio-temporal graph features using EfficientGCN blocks.

    Expected input:
        x:
            Tensor of shape [B, C_in, T, V]

            B     = batch size
            C_in  = input feature channels
            T     = number of selected keyframes
            V     = number of facial landmarks, usually 51

        Adj:
            Fixed or dynamic adjacency matrix.

            Common possible shapes:
                [V, V]
                [B, V, V]

            The exact supported shape depends on your Spatial_Graph_Layer.

    Output:
        out:
            Tensor of shape [B, stream_embedding]

            This is the graph-stream feature embedding.
    """

    def __init__(self, args):
        super(EfficientGCN, self).__init__()

        # Build the main spatio-temporal graph feature extractor.
        self.main_stream = EfficientGCN_Blocks(args, block_args=args.model_block_args)

        # The final channel size is taken from the last EfficientGCN block.
        last_channel = args.model_block_args[-1][0]

        # Convert the final graph feature map into a compact feature embedding.
        self.classifier = EfficientGCN_Classifier(args, last_channel)

        # Initialize trainable parameters.
        init_param(self.modules())

    def forward(self, x, Adj):
        """
        Forward pass of the graph stream.

        Args:
            x:
                Input landmark feature tensor with shape [B, C_in, T, V].

            Adj:
                Adjacency matrix. It can be fixed [V, V] or batch-wise
                dynamic [B, V, V], depending on Spatial_Graph_Layer support.

        Returns:
            out:
                Feature embedding with shape [B, stream_embedding].
        """

        # Extract spatio-temporal graph features.
        x = self.main_stream(x, Adj)

        # Global pooling + projection.
        out = self.classifier(x)

        # Remove spatial and temporal singleton dimensions after GAP.
        # Shape: [B, stream_embedding, 1, 1] -> [B, stream_embedding]
        out = out.squeeze(dim=2)
        out = out.squeeze(dim=2)

        return out


class EfficientGCN_Blocks(nn.Module):
    """
    A stack of EfficientGCN blocks.

    Each block contains:
        1. Spatial graph convolution layer
        2. Temporal convolution layer or layers
        3. Spatio-temporal landmark attention layer

    Block structure:
        x -> GCN -> TCN -> STLA -> output

    This corresponds to the STLAGCB-style processing described in the paper:
        STLAGCB = STLA(TCN(SGC(A, X)))

    Expected input:
        x:
            Tensor of shape [B, C_in, T, V]

        Adj:
            Adjacency matrix used by the spatial graph layer.

    Output:
        x:
            Tensor of shape [B, C_out, T_out, V]
    """

    def __init__(self, args, block_args):
        super(EfficientGCN_Blocks, self).__init__()

        # kernel_size is expected to contain:
        #   temporal_window_size: kernel size for temporal convolution
        #   max_graph_distance: graph distance for spatial graph convolution
        temporal_window_size, max_graph_distance = args.kernel_size

        # Initial input channel size.
        last_channel = args.stream_input_channel

        # Basic temporal convolution is used here.
        # Other imported temporal layers can be used if needed.
        temporal_layer = Temporal_Basic_Layer

        self.blocks = nn.ModuleList()

        for i, [output_channel, stride, depth] in enumerate(block_args):
            # Spatial graph convolution.
            # This models landmark-to-landmark relationships.
            gcn = Spatial_Graph_Layer(
                args, last_channel, output_channel, max_graph_distance
            )

            # Temporal convolution stack.
            # The first temporal layer in a block may use stride.
            # Later layers keep stride = 1.
            tcn = nn.Sequential()

            for j in range(depth):
                current_stride = stride if j == 0 else 1

                tcn.add_module(
                    f"block-{i}_tcn-{j}",
                    temporal_layer(
                        args,
                        output_channel,
                        temporal_window_size,
                        stride=current_stride,
                    ),
                )

            # Spatio-temporal landmark attention.
            # This reweights important landmarks and keyframes.
            attn = Attention_Layer(args, output_channel)

            # Update channel size for the next block.
            last_channel = output_channel

            self.blocks.append(
                nn.ModuleDict(
                    {
                        "gcn": gcn,
                        "tcn": tcn,
                        "attn": attn,
                    }
                )
            )

    def forward(self, x, Adj):
        """
        Forward pass through all EfficientGCN blocks.

        Args:
            x:
                Input feature tensor with shape [B, C, T, V].

            Adj:
                Fixed or dynamic adjacency matrix.

        Returns:
            x:
                Output feature tensor after all GCN, TCN, and attention blocks.
        """

        for block in self.blocks:
            x = block["gcn"](x, Adj)
            x = block["tcn"](x)
            x = block["attn"](x)

        return x


class EfficientGCN_Classifier(nn.Sequential):
    """
    Feature projection head for EfficientGCN.

    This module does not directly output class logits in your current design.
    Instead, it converts the final graph feature map into a compact embedding.

    Operations:
        1. Adaptive average pooling to [1, 1]
        2. Dropout
        3. 1 x 1 convolution to produce stream_embedding channels

    Input:
        x:
            Tensor of shape [B, C, T, V]

    Output:
        x:
            Tensor of shape [B, stream_embedding, 1, 1]
    """

    def __init__(self, args, curr_channel):
        super(EfficientGCN_Classifier, self).__init__()

        # Number of classes is kept here for compatibility,
        # although this classifier currently outputs embeddings.
        self.num_class = 3 if args.num_classes == "Folder" else int(args.num_classes)

        self.add_module("gap", nn.AdaptiveAvgPool2d(1))
        self.add_module("dropout", nn.Dropout(args.drop_prob, inplace=True))

        self.add_module(
            "fc", nn.Conv2d(curr_channel, args.stream_embedding, kernel_size=1)
        )


class EdgePredictor(nn.Module):
    """
    Spatio-Temporal Graph Autoencoder decoder for dynamic adjacency generation.

    This module implements the decoder described in the paper.

    The encoder first extracts latent spatio-temporal landmark features.
    Then the decoder reconstructs a dynamic adjacency matrix from those
    landmark-wise latent representations.

    Decoder formulation:

        1. Temporal average pooling:
            Z = GAP_time(H)

        2. Bilinear score matrix:
            S = Z B Z^T / sqrt(d_z)

        3. Sigmoid and symmetrization:
            A_hat = sigmoid((S + S^T) / 2)

        4. Add self-loops:
            A_tilde = A_hat + I

        5. Symmetric degree normalization:
            DA = D^{-1/2} A_tilde D^{-1/2}

    Important:
        In the paper, B is a learnable bilinear transformation matrix with
        shape [d_z, d_z], where d_z is the latent feature dimension.

        Therefore, B should not be [51, 51].
        The landmark dimension is handled by Z and Z^T.

    Expected input:
        x:
            Tensor of shape [B, C_in, T, V]

            B     = batch size
            C_in  = input channel size
            T     = number of selected keyframes
            V     = number of facial landmarks, usually 51

        Adj:
            Initial adjacency matrix used by the STGAE encoder.
            Usually this can be a complete graph or any initial graph prior.

    Output:
        DA:
            Dynamic adjacency matrix with shape [B, V, V].

            Each sample in the batch gets its own learned adjacency matrix.
    """

    def __init__(self, args):
        super(EdgePredictor, self).__init__()

        # STGAE encoder.
        # This extracts latent landmark features before adjacency reconstruction.
        self.Encoder = EfficientGCN_Blocks(args, block_args=args.ep_block_args)

        # Final latent dimension d_z.
        # Each block arg is assumed to follow:
        #     [output_channel, stride, depth]
        dz = args.ep_block_args[-1][0]

        # Learnable bilinear transformation matrix B.
        # Paper formula:
        #     S = Z B Z^T / sqrt(d_z)
        #
        # Shape:
        #     B: [d_z, d_z]
        self.B = nn.Parameter(torch.empty(dz, dz))

        # Xavier initialization is suitable for bilinear projection weights.
        nn.init.xavier_uniform_(self.B)

        # Small constant for numerical stability during degree normalization.
        self.eps = 1e-6

        # Initialize only the encoder modules.
        # B is already initialized above.
        init_param(self.Encoder.modules())

    def forward(self, x, Adj):
        """
        Generate dynamic adjacency matrix.

        Args:
            x:
                Input node feature tensor with shape [B, C_in, T, V].

            Adj:
                Initial adjacency matrix for the encoder.

        Returns:
            DA:
                Normalized dynamic adjacency matrix with shape [B, V, V].
        """

        # ---------------------------------------------------------------
        # 1. Encode spatio-temporal landmark features
        # ---------------------------------------------------------------
        # Output shape:
        #     x: [B, d_z, T, V]
        x = self.Encoder(x, Adj)

        if x.dim() != 4:
            raise ValueError(
                f"Expected encoder output with shape [B, C, T, V], "
                f"but got shape {tuple(x.shape)}."
            )

        B_size, dz, T_size, V_size = x.shape

        # ---------------------------------------------------------------
        # 2. Temporal global average pooling
        # ---------------------------------------------------------------
        # The paper applies average pooling along the temporal dimension.
        #
        # Shape:
        #     [B, d_z, T, V] -> [B, d_z, V]
        z = x.mean(dim=2)

        # Convert to landmark-wise representation.
        #
        # Shape:
        #     [B, d_z, V] -> [B, V, d_z]
        z = z.permute(0, 2, 1).contiguous()

        # Safety check.
        if z.size(-1) != self.B.size(0):
            raise ValueError(
                f"Latent feature dimension mismatch. "
                f"z has d_z={z.size(-1)}, but B has shape {tuple(self.B.shape)}."
            )

        # ---------------------------------------------------------------
        # 3. Bilinear decoder
        # ---------------------------------------------------------------
        # Paper formula:
        #     S = Z B Z^T / sqrt(d_z)
        #
        # Shapes:
        #     z:      [B, V, d_z]
        #     B:      [d_z, d_z]
        #     S:      [B, V, V]
        S = torch.einsum("bvc,cd,bwd->bvw", z, self.B, z)

        # Scaling prevents bilinear scores from growing too large when
        # the latent feature dimension increases.
        S = S / math.sqrt(dz)

        # ---------------------------------------------------------------
        # 4. Sigmoid bounding and symmetrization
        # ---------------------------------------------------------------
        # S contains raw edge scores.
        # Sigmoid maps them to [0, 1].
        #
        # Symmetrization makes the graph undirected:
        #     A_hat[i, j] = A_hat[j, i]
        A_hat = torch.sigmoid((S + S.transpose(1, 2)) / 2.0)

        # ---------------------------------------------------------------
        # 5. Add self-loops
        # ---------------------------------------------------------------
        # Self-loops allow each landmark to preserve its own information
        # during graph message passing.
        I = torch.eye(V_size, device=A_hat.device, dtype=A_hat.dtype).unsqueeze(0)

        A_tilde = A_hat + I

        # ---------------------------------------------------------------
        # 6. Symmetric degree normalization
        # ---------------------------------------------------------------
        # Degree matrix:
        #     D[i] = sum_j A_tilde[i, j]
        #
        # Normalized adjacency:
        #     DA = D^{-1/2} A_tilde D^{-1/2}
        #
        # This prevents high-degree landmarks from dominating message passing.
        degree = A_tilde.sum(dim=-1)

        degree_inv_sqrt = torch.pow(degree.clamp(min=self.eps), -0.5)

        DA = degree_inv_sqrt.unsqueeze(-1) * A_tilde * degree_inv_sqrt.unsqueeze(-2)

        return DA


def init_param(modules):
    """
    Initialize model parameters.

    Initialization strategy:
        Conv1d / Conv2d:
            Kaiming normal initialization.

        BatchNorm1d / BatchNorm2d / BatchNorm3d:
            Weight initialized to 1.
            Bias initialized to 0.

        Conv3d / Linear:
            Normal initialization with std = 0.001.

    Args:
        modules:
            Iterable of PyTorch modules, usually self.modules().
    """

    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif (
            isinstance(m, nn.BatchNorm1d)
            or isinstance(m, nn.BatchNorm2d)
            or isinstance(m, nn.BatchNorm3d)
        ):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
