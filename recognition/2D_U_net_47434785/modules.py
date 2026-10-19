import torch
from torch import nn
from torch.nn import functional as F

"""Standard 2D U-Net baseline for COMP3710 HipMRI segmentation.

Architecture reference:
    Ronneberger, O., Fischer, P., & Brox, T. (2015).
    "U-Net: Convolutional Networks for Biomedical Image Segmentation."
    https://arxiv.org/abs/1505.04597

Implementation attribution:
    PyTorch code written with ChatGPT assistance based on Figure 1 and
    Section 2 of Ronneberger et al. (2015).

    Differs from reference architecture by adding batch normalisation 
    and Billinear interpolation instead of Transposed convolution.

"""
class DoubleConv(nn.Module):
    """Two 3x3 convolution -> batch normalisation -> ReLU operations.

    Args:
        in_channels: Input feature channels.
        out_channels: Output feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return extracted features, keeping the same height and width."""
        return self.layers(x)


class Encode(nn.Module):
    """One U-Net encoder stage: max pooling followed by double convolution.

    Args:
        in_channels: Feature channels before downsampling.
        out_channels: Feature channels after double convolution.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return features with half the input height and width."""
        return self.layers(x)


class Decode(nn.Module):
    """One U-Net decoder stage: upsample, concatenate skip features, convolve.

    Args:
        in_channels: Channels arriving from the previous decoder stage.
        skip_channels: Channels in the corresponding encoder skip tensor.
        out_channels: Channels produced by this decoder stage.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Billinear Upsample of x to skip_channels size, concatenate, and return refined features."""
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """Baseline 2D U-Net (Ronneberger et al., 2015), adapted for HipMRI.

    Uses five resolution levels, four pooling stages, four decoder stages,
    encoder-decoder skip connections and a final 1x1 output convolution.

    Args:
        in_channels: Number of input channels (1 for HipMRI).
        num_classes: Number of output segmentation classes (6 for HipMRI).
        base_channels: Number of feature channels at the first level (64).

    Returns from forward:
        Tensor of raw per-class logits with shape (B, num_classes, H, W).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 6,
                 base_channels: int = 64):
        super().__init__()
        if in_channels < 1 or num_classes < 1 or base_channels < 1:
            raise ValueError("Channel counts must be positive integers.")

        c = base_channels
        # Encoder: features grow while spatial dimensions shrink.
        self.stem = DoubleConv(in_channels, c)
        self.down1 = Encode(c, 2 * c)
        self.down2 = Encode(2 * c, 4 * c)
        self.down3 = Encode(4 * c, 8 * c)
        self.down4 = Encode(8 * c, 16 * c)  # Bottleneck at 1/16 resolution.

        # Decoder: upsampling and concatenation with encoder skip features.
        self.up1 = Decode(16 * c, 8 * c, 8 * c)
        self.up2 = Decode(8 * c, 4 * c, 4 * c)
        self.up3 = Decode(4 * c, 2 * c, 2 * c)
        self.up4 = Decode(2 * c, c, c)

        # One output logit map per segmentation class.
        self.output_conv = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Segment a batch of MRI slices; return logits at input resolution."""
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)

        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        return self.output_conv(x)