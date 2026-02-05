import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=5, latent_channels=64):
        """
        Compresses HxW image to H/16 x W/16 feature map
        """
        super(Encoder, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ) # 64x1024
        
        self.layer1 = ResidualBlock(64, 64, stride=2)   # 32x512
        self.layer2 = ResidualBlock(64, 128, stride=2)  # 16x256
        self.layer3 = ResidualBlock(128, 256, stride=2) # 8x128
        self.layer4 = ResidualBlock(256, latent_channels, stride=2) # 4x64
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_channels=64, out_channels=5):
        """
        Reconstructs HxW image from H/16 x W/16 feature map
        """
        super(Decoder, self).__init__()
        
        self.deconv1 = nn.ConvTranspose2d(latent_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1) # 8x128
        self.bn1 = nn.BatchNorm2d(256)
        
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1) # 16x256
        self.bn2 = nn.BatchNorm2d(128)
        
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1) # 32x512
        self.bn3 = nn.BatchNorm2d(64)
        
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # 64x1024
        self.bn4 = nn.BatchNorm2d(32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = self.final_conv(x)
        return x # Range, x, y, z, remission

class RangeCompressionModel(nn.Module):
    def __init__(self):
        super(RangeCompressionModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x, noise_std=0.0):
        # x: [B, 5, H, W]
        latent = self.encoder(x)
        
        # Add quantization-aware noise or transmission noise
        if self.training and noise_std > 0.0:
            noise = torch.randn_like(latent) * noise_std
            latent = latent + noise
            
        recon = self.decoder(latent)
        return recon, latent

if __name__ == "__main__":
    # Test dims
    model = RangeCompressionModel()
    dummy_input = torch.randn(1, 5, 64, 1024)
    recon, latent = model(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Latent: {latent.shape}")
    print(f"Recon: {recon.shape}")
