# from torch import nn
import torch
import torch.nn as nn

class kirsch_mask_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding='same', bias=False)

        na= torch.tensor([[-3.0,-3.0,5.0],[-3.0,0.0,5.0],[-3.0,-3.0,5.0]]).unsqueeze(0) # 1, 3, 3
        nwa= torch.tensor([[-3.0,5.0,5.0],[-3.0,0.0,5.0],[-3.0,-3.0,-3.0]]).unsqueeze(0)
        wa= torch.tensor([[5.0,5.0,5.0],[-3.0,0.0,-3.0],[-3.0,-3.0,-3.0]]).unsqueeze(0)
        swa= torch.tensor([[5.0,5.0,-3.0],[5.0,0.0,-3.0],[-3.0,-3.0,-3.0]]).unsqueeze(0)
        sa= torch.tensor([[5.0,-3.0,-3.0],[5.0,0.0,-3.0],[5.0,-3.0,-3.0]]).unsqueeze(0)
        sea= torch.tensor([[-3.0,-3.0,-3.0],[5.0,0.0,-3.0],[5.0,5.0,-3.0]]).unsqueeze(0)
        ea= torch.tensor([[-3.0,-3.0,-3.0],[-3.0,0.0,-3.0],[5.0,5.0,5.0]]).unsqueeze(0)
        nea= torch.tensor([[-3.0,-3.0,-3.0],[-3.0,0.0,5.0],[-3.0,5.0,5.0]]).unsqueeze(0)

        k = torch.cat([na, nwa, wa, swa, sa, sea, ea, nea], 0)
        k = k.unsqueeze(1) # k.shape
        self.filter.weight = nn.Parameter(k, requires_grad=False)

    def forward(self, img):
        img = torch.unsqueeze(img, dim=0) # 1, 480, 640
        img = torch.unsqueeze(img, dim=1) # 1, 1, 480, 640
        x = self.filter(img) # 1, 8, 480, 640
        x = x.permute(0, 2, 3, 1)
        x = torch.squeeze(x, dim=0)
        return 8*x.argmax(axis=2) + x.argmin(axis=2)