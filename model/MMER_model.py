import torch
import torch.nn as nn
from model.st_gcn import EfficientGCN, EdgePredictor
from data_utilz.graphs import Graph
from model.unite_models import vit_model, lstm_model, attentive_feature_fusion, Multi_attentive_feature_fusion



class ConvBlock(nn.Module):
    def __init__(self, **kwargs):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.BatchNorm2d(kwargs["out_channels"]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DWConv(nn.Module):
    def __init__(self, **kwargs):
        super(DWConv, self).__init__()

        self.block = nn.Sequential(
            ConvBlock(in_channels=kwargs["in_channels"],
                      out_channels=kwargs["in_channels"],
                      kernel_size=kwargs["kernel_size"],
                      padding=kwargs["kernel_size"] // 2,
                      groups=kwargs["in_channels"],
                      bias=False),
            ConvBlock(in_channels=kwargs["in_channels"],
                      out_channels=kwargs["out_channels"],
                      kernel_size=1,
                      bias=False),
            nn.Flatten(start_dim=2)
        )

    def forward(self, x):
        return self.block(x)


class facial_graph_stream(nn.Module):
    """
    Facial Graph Stream Module

    Parameters:
    - args: A namespace containing various configuration arguments, including the number of landmarks and other model-specific settings.
    - device (torch.device): The device (CPU or GPU) on which the model and data will be processed.

    Attributes:
    - num_landmarks (int): The number of facial landmarks used in the graph. This is set to 51 by default.
    - A (torch.Tensor): The initial adjacency matrix representing the connections between facial landmarks. Initialized to ones and moved to the specified device.
    - gsae (EdgePredictor): A graph structure autoencoder used to predict the dynamic adjacency matrix.
    - stfgn (EfficientGCN): A spatial-temporal landmark attention-based graph convolutional network used to process the facial landmarks with the dynamic adjacency matrix.
    """
    def __init__(self, args, device):
        super(facial_graph_stream, self).__init__()

        self.num_landmarks = args.num_landmarks
        self.num_landmarks = 51
        self.A = torch.ones((self.num_landmarks, self.num_landmarks)).float().to(device)
        self.gsae = EdgePredictor(args)
        self.stfgn = EfficientGCN(args)

    def forward(self, X):
        DA = self.gsae(X, self.A)
        DA.fill_diagonal_(1.0)
        fgs_out = self.stfgn(X, DA)
        return fgs_out

class visual_stream(nn.Module):
    """
    Visual Stream Module

    Parameters:
    - args: A namespace containing various configuration arguments, including the number of frames and stream embedding size.

    Attributes:
    - num_frames (int): The number of frames in the input sequence.
    - pretrained_vit (nn.Module): A pretrained Vision Transformer model used for extracting spatial features from each frame.
    - lstm (nn.Module): A Bidirectional LSTM model with 3 layers, used to capture temporal dependencies across frames.
    """
    def __init__(self, args):
        super(visual_stream, self).__init__()

        self.num_frames = args.num_frames
        self.pretrained_vit = vit_model(args.stream_embedding)
        self.lstm = lstm_model(num_layers=3, input_size=128, hidden_size=64, num_frames=self.num_frames)

    def forward(self, stldn):
        x = torch.unsqueeze(stldn, dim=1)
        N, C, S, H, W = x.size()
        x = x.view(N * S, C, H, W)
        x = x.repeat(1, 3, 1, 1)
        vit_out = self.pretrained_vit(x)
        vit_out = vit_out.view(N, S, -1)
        bilstm_out = self.lstm(vit_out)
        return bilstm_out

class MMER_model(nn.Module):
    def __init__(self, args, device, num_classes):
        super(MMER_model, self).__init__()

        # num_class = 3 if args.num_classes == 'Folder' else int(args.num_classes)

        if num_classes == 'Folder':
            self.num_class = 3
        else:
            self.num_class = int(num_classes)
        self.final_embadding = 128  # self.num_class
        self.device = device

        self.fgs = facial_graph_stream(args, device).to(device)
        self.vs = visual_stream(args).to(device)

        self.num_landmarks = args.num_landmarks
        self.num_frames = args.num_frames
        self.num_features = args.num_features
        self.feature_size = (self.num_frames - 1) * self.num_landmarks  # T*C n(3*51=153)
        self.stream_embedding = args.stream_embedding
        # self.feature_size = (self.num_frames-1)*self.num_landmarks  # T*C n(3*51=153)
        kwargs = {'in_channels': self.feature_size, 'out_channels': self.feature_size, 'stride': 1, 'kernel_size': 3}
        self.appearance_processing = DWConv(**kwargs)  # input = 153*7*7 | output = (T*V)*C -> 153, 49

        self.aff = attentive_feature_fusion(in_channel=self.stream_embedding, out_channel=self.stream_embedding)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.stream_embedding, self.num_class),
            nn.Softmax(dim=1)
        )

    def forward(self, X, stldn_seq):
        X = self.appearance_processing(X)
        X = X.view(-1, self.num_features, self.num_frames - 1, self.num_landmarks)
        fgs_out = self.fgs (X)
        vs_out = self.vs(stldn_seq)
        aff_out = self.aff(fgs_out, vs_out)
        output = self.classifier(aff_out)
        return output





