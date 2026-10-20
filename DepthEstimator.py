import torch
import torch.nn as nn
import timm

class DepthEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("hf-hub:timm/mobilenetv4_conv_medium.e250_r384_in12k", pretrained = True, features_only = True) 
        self.depth_head = DepthUNet()
        cfg = timm.data.resolve_model_data_config(self.encoder)
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1,3,1,1))
    def forward(self, x): # (B, 3, 224, 224)
        x = (x - self.mean)/self.std
        out = self.encoder(x)
        return self.depth_head(out)

class DepthUNet(nn.Module):
    def __init__(self): 
        super().__init__()
        self.init_conv = nn.Conv2d(960, 96, 1, 1, 0)
        self.init_bn = nn.BatchNorm2d(96)
        self.relu = nn.ReLU()
        self.up1_pre = ConvBlock(96, 96)
        self.up1_post = ConvBlock(256, 96)

        self.up2_pre = ConvBlock(96, 64)
        self.up2_post = ConvBlock(144, 64)

        self.up3_pre = ConvBlock(64, 32)
        self.up3_post = ConvBlock(80, 32)

        self.up4_pre = ConvBlock(32,16)
        self.up4_post = ConvBlock(48,16)

        self.final_conv = nn.Conv2d(16,1,(3,3),1,1)

#If B = 15 (example) MobileNet outputs these shapes
#torch.Size([15, 32, 112, 112])
#torch.Size([15, 48, 56, 56])
#torch.Size([15, 80, 28, 28])
#torch.Size([15, 160, 14, 14])
#torch.Size([15, 960, 7, 7])
    def forward(self, x):
        out = self.relu(self.init_bn(self.init_conv(x[-1]))) # (B, 96, 7, 7)
        out = torch.nn.functional.interpolate(out, scale_factor = 2, mode = "bilinear") # (B, 96, 14,14)
        out = torch.cat([x[-2], out], dim = 1) # (B, 256, 14, 14)
        out = self.up1_post(out) # (B, 96, 14, 14)

        out = self.up2_pre(out)
        out = torch.nn.functional.interpolate(out, scale_factor = 2.0, mode = "bilinear")
        out = torch.cat([x[-3], out], dim = 1) # B, 144, 28, 28
        out = self.up2_post(out) #B, 64, 28, 28

        out = self.up3_pre(out) #32, 28, 28
        out = torch.nn.functional.interpolate(out, scale_factor = 2.0, mode = "bilinear") # 32, 56, 56
        out = torch.cat([x[-4], out], dim = 1) # 80, 56, 56
        out = self.up3_post(out) # 32, 56, 56

        out = self.up4_pre(out) #16, 56, 56
        out = torch.nn.functional.interpolate(out, scale_factor = 2.0, mode = "bilinear") # 16, 112, 112
        out = torch.cat([x[-5], out], dim = 1) #48, 112, 112
        out = self.up4_post(out) # 16, 112, 112
        out = torch.nn.functional.interpolate(out, scale_factor = 2, mode = "bilinear") # 1, 224, 224
        out = self.final_conv(out) # 1, 112, 112

        return out


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.conv = nn.Conv2d(cin, cout, (3,3), 1, 1)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

