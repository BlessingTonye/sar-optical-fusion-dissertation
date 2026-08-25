import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# U-Net (standard, single-encoder configurations)

def build_unet(encoder_name, in_channels, n_classes):
    """
    Builds a U-Net model with the given encoder, matching this study's
    from-scratch initialization convention.

    Args:
        encoder_name (str): encoder backbone.
        in_channels (int): number of input channels.
        n_classes (int): number of output classes.

    Returns:
        nn.Module: constructed U-Net model.
    """
    return smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=in_channels, classes=n_classes)


# Middle Fusion (custom dual-encoder architecture)

class MiddleFusionBlock(nn.Module):
    """
    Merges S1 and S2 feature maps at one encoder stage, projecting
    the concatenated features back to the original channel count.

    Args:
        channels (int): number of channels at this encoder stage.
    """
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),  # 1x1: channel blending only, no spatial mixing
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_s1, feat_s2):
        """
        Args:
            feat_s1 (Tensor): S1 feature map at this encoder stage.
            feat_s2 (Tensor): S2 feature map at this encoder stage.

        Returns:
            Tensor: fused feature map, same channel count as input.
        """
        combined = torch.cat([feat_s1, feat_s2], dim=1)
        return self.fuse(combined)


class DecoderBlock(nn.Module):
    """
    One decoder stage: upsamples the input, optionally merges it with
    a matching-resolution encoder skip connection, then refines the
    result with two convolutional passes.

    Args:
        in_channels (int): channels in the incoming feature map.
        skip_channels (int): channels in the skip connection (0 if none).
        out_channels (int): channels in the output feature map.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        """
        Args:
            x (Tensor): feature map from the previous decoder stage.
            skip (Tensor, optional): matching-resolution encoder feature map.

        Returns:
            Tensor: refined, upsampled feature map.
        """
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MiddleFusionUNet(nn.Module):
    """
    Custom dual-encoder architecture for feature-level (middle) fusion.
    S1 and S2 are processed by independent encoders, fused via
    MiddleFusionBlock at each stage, then passed through a shared
    decoder to produce per-pixel class scores.

    Args:
        n_s1_bands (int): number of S1 input channels.
        n_s2_bands (int): number of S2 input channels.
        n_classes (int): number of output classes.
    """
    def __init__(self, n_s1_bands, n_s2_bands, n_classes):
        super().__init__()
        self.encoder_s1 = smp.encoders.get_encoder("resnet34", in_channels=n_s1_bands, weights=None)
        self.encoder_s2 = smp.encoders.get_encoder("resnet34", in_channels=n_s2_bands, weights=None)

        self.fusion_blocks = nn.ModuleList([
            MiddleFusionBlock(64), MiddleFusionBlock(64), MiddleFusionBlock(128),
            MiddleFusionBlock(256), MiddleFusionBlock(512),
        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_channels=512, skip_channels=256, out_channels=256),
            DecoderBlock(in_channels=256, skip_channels=128, out_channels=128),
            DecoderBlock(in_channels=128, skip_channels=64,  out_channels=64),
            DecoderBlock(in_channels=64,  skip_channels=64,  out_channels=32),
            DecoderBlock(in_channels=32,  skip_channels=0,   out_channels=16),
        ])

        self.segmentation_head = nn.Conv2d(16, n_classes, kernel_size=3, padding=1)

    def forward(self, x_s1, x_s2):
        """
        Args:
            x_s1 (Tensor): S1 input batch, shape (batch, n_s1_bands, H, W).
            x_s2 (Tensor): S2 input batch, shape (batch, n_s2_bands, H, W).

        Returns:
            Tensor: per-pixel class scores, shape (batch, n_classes, H, W).
        """
        s1_features = self.encoder_s1(x_s1)
        s2_features = self.encoder_s2(x_s2)

        fused = [self.fusion_blocks[i](s1_features[i + 1], s2_features[i + 1]) for i in range(5)]  # skip stage 0 (raw input)

        x = fused[4]
        x = self.decoder_blocks[0](x, fused[3])
        x = self.decoder_blocks[1](x, fused[2])
        x = self.decoder_blocks[2](x, fused[1])
        x = self.decoder_blocks[3](x, fused[0])
        x = self.decoder_blocks[4](x, None)

        return self.segmentation_head(x)


# SegFormer (custom implementation)

class MLP(nn.Module):
    """
    Projects a single encoder stage's features to a common embedding
    dimension, the basic building block of SegFormer's decoder head.

    Args:
        in_channels (int): number of channels in this encoder stage.
        embed_dim (int): shared embedding dimension across all stages.
    """
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): feature map, shape (batch, channels, H, W).

        Returns:
            Tensor: projected feature map, shape (batch, embed_dim, H, W).
        """
        batch, channels, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(batch, -1, h, w)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer's lightweight All-MLP decoder: projects every encoder
    stage to the same embedding size, upsamples all stages to a
    common resolution, concatenates, and fuses with one final layer.

    Args:
        encoder_channels (list): output channels of each encoder stage used.
        embed_dim (int): common projection dimension for all stages.
        n_classes (int): number of segmentation classes.
    """
    def __init__(self, encoder_channels, embed_dim=256, n_classes=4):
        super().__init__()
        self.mlps = nn.ModuleList([MLP(c, embed_dim) for c in encoder_channels])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(encoder_channels), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(embed_dim, n_classes, kernel_size=1)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): feature maps from encoder stages 2-5.

        Returns:
            Tensor: per-pixel class scores, at the highest-resolution stage's size.
        """
        target_size = features[0].shape[2:]
        projected = []
        for feat, mlp in zip(features, self.mlps):
            x = mlp(feat)
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            projected.append(x)

        x = torch.cat(projected, dim=1)
        x = self.fuse(x)
        return self.classifier(x)


class SegFormer(nn.Module):
    """
    Full SegFormer architecture: MixVisionTransformer encoder paired
    with its native All-MLP decoder head, in contrast to this study's
    other transformer configurations, which pair the same encoder
    with a U-Net decoder.

    Args:
        encoder_name (str): MiT variant, e.g. 'mit_b0', 'mit_b2'.
        in_channels (int): number of input channels.
        n_classes (int): number of segmentation classes.
    """
    def __init__(self, encoder_name="mit_b0", in_channels=6, n_classes=4):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=in_channels, weights=None)
        stage_channels = self.encoder.out_channels[2:]  # skip input stage and the zero-channel stage
        self.head = SegFormerHead(stage_channels, embed_dim=256, n_classes=n_classes)

    def forward(self, x):
        """
        Args:
            x (Tensor): input batch, shape (batch, in_channels, H, W).

        Returns:
            Tensor: per-pixel class scores, shape (batch, n_classes, H, W).
        """
        input_size = x.shape[2:]
        features = self.encoder(x)[2:]  # match the same skip
        out = self.head(features)
        return F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)