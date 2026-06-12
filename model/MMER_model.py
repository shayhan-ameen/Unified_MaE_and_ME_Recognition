import torch
import torch.nn as nn
from model.st_gcn import EdgePredictor, EfficientGCN
from model.unite_models import (
    attentive_feature_fusion,
    lstm_model,
    vit_model,
)


class ConvBlock(nn.Module):
    """
    Basic convolution block.

    This block applies:

        Conv2d -> BatchNorm2d -> ReLU

    It is used inside DWConv to build a depthwise separable convolution module.

    Expected input shape:
        x: [B, C_in, H, W]

    Output shape:
        x: [B, C_out, H_out, W_out]

    Args:
        **kwargs:
            Keyword arguments passed directly to nn.Conv2d.
            Must include:
                in_channels
                out_channels
                kernel_size

            Optional:
                stride
                padding
                groups
                bias
    """

    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.BatchNorm2d(kwargs["out_channels"]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Input tensor with shape [B, C_in, H, W].

        Returns:
            Tensor with shape [B, C_out, H_out, W_out].
        """

        return self.block(x)


class DWConv(nn.Module):
    """
    Depthwise separable convolution block for landmark patch processing.

    This module processes local STLDN patches around facial landmarks.

    It applies:

        1. Depthwise convolution:
            Each input channel is convolved independently.

        2. Pointwise convolution:
            A 1 x 1 convolution mixes information across channels.

        3. Flatten:
            Spatial patch dimensions are flattened into a feature vector.

    Expected input shape:
        x: [B, C_in, S, S]

        B     = batch size
        C_in  = usually T * V
                T = number of STLDN frames
                V = number of landmarks
        S     = local patch size, for example 7

    Output shape:
        x: [B, C_out, S * S]

    In your framework:
        C_in  = (num_frames - 1) * num_landmarks
        C_out = (num_frames - 1) * num_landmarks

    After this module, the output is reshaped to:

        [B, num_features, num_frames - 1, num_landmarks]

    where:
        num_features = S * S
    """

    def __init__(self, **kwargs):
        super(DWConv, self).__init__()

        self.block = nn.Sequential(
            # Depthwise convolution.
            ConvBlock(
                in_channels=kwargs["in_channels"],
                out_channels=kwargs["in_channels"],
                kernel_size=kwargs["kernel_size"],
                padding=kwargs["kernel_size"] // 2,
                groups=kwargs["in_channels"],
                bias=False,
            ),
            # Pointwise convolution.
            ConvBlock(
                in_channels=kwargs["in_channels"],
                out_channels=kwargs["out_channels"],
                kernel_size=1,
                bias=False,
            ),
            # Flatten spatial patch dimensions.
            # [B, C, S, S] -> [B, C, S*S]
            nn.Flatten(start_dim=2),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x:
                Landmark patch tensor with shape [B, C_in, S, S].

        Returns:
            Tensor with shape [B, C_out, S*S].
        """

        return self.block(x)


class facial_graph_stream(nn.Module):
    """
    Facial Graph Stream.

    This stream corresponds to the graph-based branch of the proposed model.

    It performs two main steps:

        1. Dynamic adjacency generation using STGAE:
            DA = EdgePredictor(X, A)

        2. Spatio-temporal graph feature extraction using STFGN:
            fgs_out = EfficientGCN(X, DA)

    Input:
        X:
            Landmark feature tensor with shape:

                [B, F, T, V]

            B = batch size
            F = landmark feature dimension, for example 49 from 7 x 7 patches
            T = number of selected STLDN frames, usually num_frames - 1
            V = number of facial landmarks, usually 51

    Output:
        fgs_out:
            Facial graph stream embedding with shape:

                [B, stream_embedding]

    Important:
        The updated EdgePredictor returns a batch-wise dynamic adjacency matrix:

            DA.shape = [B, V, V]

        Therefore, Spatial_Graph_Layer must support batched adjacency.
    """

    def __init__(self, args, device):
        super(facial_graph_stream, self).__init__()

        # Number of facial landmarks.
        # The paper uses 51 inner-face landmarks.
        self.num_landmarks = int(getattr(args, "num_landmarks", 51))

        # Initial adjacency matrix.
        # This is used as the input graph prior for the STGAE encoder.
        #
        # register_buffer makes A part of the module state, but not trainable.
        # It also moves automatically when model.to(device) is called.
        self.register_buffer(
            "A",
            torch.ones(self.num_landmarks, self.num_landmarks, dtype=torch.float32),
        )

        # Spatio-Temporal Graph Autoencoder.
        # Generates dynamic adjacency matrix DA.
        self.gsae = EdgePredictor(args)

        # Spatio-Temporal Facial Graph Network.
        # Extracts graph-based facial appearance and motion features.
        self.stfgn = EfficientGCN(args)

    def forward(self, X):
        """
        Forward pass.

        Args:
            X:
                Landmark feature tensor with shape [B, F, T, V].

        Returns:
            fgs_out:
                Graph stream feature embedding with shape [B, stream_embedding].
        """

        if X.dim() != 4:
            raise ValueError(
                f"Expected X with shape [B, F, T, V], but got {tuple(X.shape)}."
            )

        B, F, T, V = X.shape

        if V != self.num_landmarks:
            raise ValueError(
                f"Landmark mismatch. Model expects V={self.num_landmarks}, "
                f"but input has V={V}."
            )

        # Generate dynamic adjacency matrix.
        #
        # Expected DA shape:
        #     [B, V, V]
        #
        # Note:
        #     Do not use DA.fill_diagonal_(1.0) here.
        #     The updated EdgePredictor already adds self-loops and applies
        #     symmetric degree normalization.
        DA = self.gsae(X, self.A)

        if DA.dim() != 3:
            raise ValueError(
                f"Expected DA with shape [B, V, V], but got {tuple(DA.shape)}."
            )

        if DA.size(0) != B or DA.size(1) != V or DA.size(2) != V:
            raise ValueError(
                f"DA shape mismatch. Expected [B, V, V] = [{B}, {V}, {V}], "
                f"but got {tuple(DA.shape)}."
            )

        # Extract graph stream features using the learned dynamic adjacency.
        fgs_out = self.stfgn(X, DA)

        return fgs_out


class visual_stream(nn.Module):
    """
    Visual Stream.

    This stream processes the STLDN image sequence.

    It performs:

        1. ViT-based spatial feature extraction from each STLDN frame.
        2. BiLSTM-based temporal modeling across the STLDN sequence.

    Input:
        stldn:
            STLDN sequence with shape:

                [B, S, H, W]

            B = batch size
            S = number of STLDN frames, usually num_frames - 1
            H = image height
            W = image width

    Internal processing:
        1. Add channel dimension:

            [B, S, H, W] -> [B, 1, S, H, W]

        2. Merge batch and sequence dimensions:

            [B, 1, S, H, W] -> [B*S, 1, H, W]

        3. Repeat single channel to RGB:

            [B*S, 1, H, W] -> [B*S, 3, H, W]

        4. Apply ViT:

            [B*S, 3, H, W] -> [B*S, D]

        5. Restore sequence:

            [B*S, D] -> [B, S, D]

        6. Apply BiLSTM:

            [B, S, D] -> [B, stream_embedding]

    Output:
        bilstm_out:
            Visual stream embedding with shape [B, stream_embedding].
    """

    def __init__(self, args):
        super(visual_stream, self).__init__()

        self.num_frames = args.num_frames

        # ViT extracts spatial features from each STLDN frame.
        self.pretrained_vit = vit_model(args.stream_embedding)

        # BiLSTM models temporal dependencies among ViT frame embeddings.
        #
        # Current setting:
        #     input_size = 128
        #     hidden_size = 64
        #     bidirectional output usually becomes 2 * hidden_size = 128
        self.lstm = lstm_model(
            num_layers=3,
            input_size=128,
            hidden_size=64,
            num_frames=self.num_frames,
        )

    def forward(self, stldn):
        """
        Forward pass.

        Args:
            stldn:
                STLDN image sequence with shape [B, S, H, W].

        Returns:
            bilstm_out:
                Visual stream feature embedding with shape [B, stream_embedding].
        """

        if stldn.dim() != 4:
            raise ValueError(
                f"Expected stldn with shape [B, S, H, W], but got {tuple(stldn.shape)}."
            )

        # Add image channel dimension.
        # [B, S, H, W] -> [B, 1, S, H, W]
        x = torch.unsqueeze(stldn, dim=1)

        B, C, S, H, W = x.size()

        # Merge batch and sequence dimensions so ViT can process each frame.
        # [B, 1, S, H, W] -> [B*S, 1, H, W]
        x = x.view(B * S, C, H, W)

        # Convert grayscale STLDN image to pseudo-RGB for ViT.
        # [B*S, 1, H, W] -> [B*S, 3, H, W]
        x = x.repeat(1, 3, 1, 1)

        # Spatial feature extraction using ViT.
        # Expected output:
        #     [B*S, D]
        vit_out = self.pretrained_vit(x)

        # Restore temporal sequence.
        # [B*S, D] -> [B, S, D]
        vit_out = vit_out.view(B, S, -1)

        # Temporal modeling using BiLSTM.
        bilstm_out = self.lstm(vit_out)

        return bilstm_out


class MMER_model(nn.Module):
    """
    Unified Macro and Micro Expression Recognition model.

    This model combines two complementary streams:

        1. Facial Graph Stream:
            Uses landmark patch features and dynamic graph modeling.

        2. Visual Stream:
            Uses STLDN images with ViT and BiLSTM.

    The two stream embeddings are fused using Attentive Feature Fusion.

    Inputs:
        X:
            Landmark-centered STLDN patches.

            Expected shape before appearance processing:

                [B, (T * V), S, S]

            B = batch size
            T = number of STLDN frames, usually num_frames - 1
            V = number of landmarks
            S = patch size, for example 7

        stldn_seq:
            STLDN image sequence.

            Expected shape:

                [B, T, H, W]

    Internal processing of X:
        1. DWConv:

            [B, T*V, S, S] -> [B, T*V, S*S]

        2. Reshape:

            [B, T*V, S*S] -> [B, S*S, T, V]

        3. Feed to Facial Graph Stream:

            [B, F, T, V] -> [B, stream_embedding]

    Fusion:
        fgs_out = Facial Graph Stream output
        vs_out  = Visual Stream output

        aff_out = attentive_feature_fusion(fgs_out, vs_out)

    Output:
        output:
            Class probabilities with shape [B, num_class]

    Important training note:
        Your classifier currently ends with Softmax.

        If you train using nn.CrossEntropyLoss, remove Softmax because
        CrossEntropyLoss expects raw logits.

        If you train using probabilities or another custom loss, keeping
        Softmax may be acceptable.
    """

    def __init__(self, args, device, num_classes):
        super(MMER_model, self).__init__()

        # Number of classes.
        if num_classes == "Folder":
            self.num_class = 3
        else:
            self.num_class = int(num_classes)

        self.final_embadding = 128
        self.device = device

        # Two-stream architecture.
        self.fgs = facial_graph_stream(args, device).to(device)
        self.vs = visual_stream(args).to(device)

        # Dataset and feature configuration.
        self.num_landmarks = int(args.num_landmarks)
        self.num_frames = int(args.num_frames)
        self.num_features = int(args.num_features)

        # STLDN sequence length used by graph stream.
        # Usually STLDN is generated from consecutive frame triplets,
        # so the graph stream uses num_frames - 1.
        self.graph_num_frames = self.num_frames - 1

        # Number of landmark-patch channels before DWConv.
        #
        # Example:
        #     graph_num_frames = 3
        #     num_landmarks = 51
        #     feature_size = 3 * 51 = 153
        self.feature_size = self.graph_num_frames * self.num_landmarks

        self.stream_embedding = args.stream_embedding

        # Appearance processing for landmark patches.
        #
        # Input:
        #     [B, feature_size, patch_size, patch_size]
        #
        # Output:
        #     [B, feature_size, patch_size * patch_size]
        kwargs = {
            "in_channels": self.feature_size,
            "out_channels": self.feature_size,
            "stride": 1,
            "kernel_size": 3,
        }

        self.appearance_processing = DWConv(**kwargs)

        # Attentive Feature Fusion.
        # It fuses graph-stream and visual-stream embeddings.
        self.aff = attentive_feature_fusion(
            in_channel=self.stream_embedding,
            out_channel=self.stream_embedding,
        )

        # Final classifier.
        #
        # Input:
        #     [B, 2 * stream_embedding]
        #
        # Output:
        #     [B, num_class]
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.stream_embedding, self.num_class),
            nn.Softmax(dim=1),
        )

    def forward(self, X, stldn_seq):
        """
        Forward pass of the unified model.

        Args:
            X:
                Landmark-centered STLDN patch tensor.

                Expected shape:
                    [B, (T * V), S, S]

                where:
                    T = num_frames - 1
                    V = num_landmarks
                    S = patch size

            stldn_seq:
                STLDN image sequence.

                Expected shape:
                    [B, T, H, W]

        Returns:
            output:
                Class probability tensor with shape [B, num_class].
        """

        if X.dim() != 4:
            raise ValueError(
                f"Expected X with shape [B, T*V, S, S], but got {tuple(X.shape)}."
            )

        if stldn_seq.dim() != 4:
            raise ValueError(
                f"Expected stldn_seq with shape [B, T, H, W], "
                f"but got {tuple(stldn_seq.shape)}."
            )

        B, C, H, W = X.shape

        if C != self.feature_size:
            raise ValueError(
                f"X channel mismatch. Expected C={self.feature_size} "
                f"because feature_size=(num_frames-1)*num_landmarks, "
                f"but got C={C}."
            )

        # ------------------------------------------------------------
        # 1. Landmark patch appearance processing
        # ------------------------------------------------------------
        # Input:
        #     [B, T*V, S, S]
        #
        # Output:
        #     [B, T*V, F]
        #
        # where:
        #     F = S*S = num_features
        X = self.appearance_processing(X)

        # Validate the flattened patch feature size.
        if X.size(-1) != self.num_features:
            raise ValueError(
                f"Flattened patch feature mismatch. Expected num_features="
                f"{self.num_features}, but got {X.size(-1)}."
            )

        # ------------------------------------------------------------
        # 2. Reshape for graph stream
        # ------------------------------------------------------------
        # Current:
        #     [B, T*V, F]
        #
        # Target:
        #     [B, F, T, V]
        X = X.view(
            -1,
            self.num_features,
            self.graph_num_frames,
            self.num_landmarks,
        )

        # ------------------------------------------------------------
        # 3. Facial Graph Stream
        # ------------------------------------------------------------
        fgs_out = self.fgs(X)

        # ------------------------------------------------------------
        # 4. Visual Stream
        # ------------------------------------------------------------
        vs_out = self.vs(stldn_seq)

        # ------------------------------------------------------------
        # 5. Attentive Feature Fusion
        # ------------------------------------------------------------
        aff_out = self.aff(fgs_out, vs_out)

        # ------------------------------------------------------------
        # 6. Classification
        # ------------------------------------------------------------
        output = self.classifier(aff_out)

        return output
