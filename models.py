import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# U-Net (single-encoder architecture)

def build_unet(encoder_name, in_channels, n_classes):
    """
    Builds a U-Net model with the specified encoder. Encoder weights
    are not pretrained, following the study's from scratch training
    approach.

    Args:
        encoder_name (str): Name of the encoder backbone.
        in_channels (int): Number of input channels.
        n_classes (int): Number of output classes.

    Returns:
        nn.Module: Constructed U-Net model.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=n_classes
    )


# Middle Fusion (dual-encoder architecture)

class MiddleFusionBlock(nn.Module):
    """
    Merges SAR and optical feature maps at one encoder stage,
    projecting the concatenated features back to the original
    channel count.

    Args:
        channels (int): Number of channels at this encoder stage.
    """
    def __init__(self, channels):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False
            ),  
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_sar, feat_optical):
        """
        Args:
            feat_sar (Tensor): SAR feature map.
            feat_optical (Tensor): Optical feature map.

        Returns:
            Tensor: Fused feature map with the original channel count.
        """
        combined = torch.cat([feat_sar, feat_optical], dim=1)

        return self.fuse(combined)


class DecoderBlock(nn.Module):
    """
    One decoder stage that upsamples the input, combines it with a
    matching-resolution encoder skip connection when available, and
    refines the result using two convolutional operations.

    Args:
        in_channels (int): Channels in the incoming feature map.
        skip_channels (int): Channels in the skip connection.
        out_channels (int): Channels in the output feature map.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode='nearest'
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        """
        Args:
            x (Tensor): Feature map from the previous decoder stage.
            skip (Tensor, optional): Matching encoder feature map.

        Returns:
            Tensor: Refined and upsampled feature map.
        """
        x = self.upsample(x)

        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class MiddleFusionUNet(nn.Module):
    """
    Dual-encoder architecture for feature-level fusion.
    SAR and optical data are processed by separate ResNet34 encoders.
    Corresponding encoder features are fused at each stage and passed
    through a shared decoder to produce per-pixel class scores.

    Args:
        n_sar_bands (int): Number of SAR input channels.
        n_optical_bands (int): Number of optical input channels.
        n_classes (int): Number of output classes.
    """
    def __init__(self, n_sar_bands, n_optical_bands, n_classes):
        super().__init__()

        self.encoder_sar = smp.encoders.get_encoder(
            "resnet34",
            in_channels=n_sar_bands,
            weights=None
        )

        self.encoder_optical = smp.encoders.get_encoder(
            "resnet34",
            in_channels=n_optical_bands,
            weights=None
        )

        self.fusion_blocks = nn.ModuleList([
            MiddleFusionBlock(64),
            MiddleFusionBlock(64),
            MiddleFusionBlock(128),
            MiddleFusionBlock(256),
            MiddleFusionBlock(512),
        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                in_channels=512,
                skip_channels=256,
                out_channels=256
            ),
            DecoderBlock(
                in_channels=256,
                skip_channels=128,
                out_channels=128
            ),
            DecoderBlock(
                in_channels=128,
                skip_channels=64,
                out_channels=64
            ),
            DecoderBlock(
                in_channels=64,
                skip_channels=64,
                out_channels=32
            ),
            DecoderBlock(
                in_channels=32,
                skip_channels=0,
                out_channels=16
            ),
        ])

        self.segmentation_head = nn.Conv2d(
            16,
            n_classes,
            kernel_size=3,
            padding=1
        )

    def forward(self, x_sar, x_optical):
        """
        Args:
            x_sar (Tensor): SAR input batch.
            x_optical (Tensor): Optical input batch.

        Returns:
            Tensor: Per-pixel class scores.
        """
        sar_features = self.encoder_sar(x_sar)
        optical_features = self.encoder_optical(x_optical)

        fused = [
            self.fusion_blocks[i](
                sar_features[i + 1],
                optical_features[i + 1]
            )
            for i in range(5)
        ]

        x = fused[4]
        x = self.decoder_blocks[0](x, fused[3])
        x = self.decoder_blocks[1](x, fused[2])
        x = self.decoder_blocks[2](x, fused[1])
        x = self.decoder_blocks[3](x, fused[0])
        x = self.decoder_blocks[4](x, None)

        return self.segmentation_head(x)


# SegFormer architecture

class MLP(nn.Module):
    """
    Projects features from one encoder stage to a common embedding
    dimension for the SegFormer decoder head.

    Args:
        in_channels (int): Number of channels in the encoder stage.
        embed_dim (int): Common embedding dimension.
    """
    def __init__(self, in_channels, embed_dim):
        super().__init__()

        self.proj = nn.Linear(
            in_channels,
            embed_dim
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Encoder feature map.

        Returns:
            Tensor: Projected feature map.
        """
        batch, _, h, w = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.transpose(1, 2).reshape(
            batch,
            -1,
            h,
            w
        )

        return x


class SegFormerHead(nn.Module):
    """
    Implements the lightweight All-MLP SegFormer decoder. Features
    from the encoder stages are projected to a common embedding
    dimension, resized to the same spatial resolution, concatenated,
    and fused before pixel-level classification.

    Args:
        encoder_channels (list): Channels from the encoder stages.
        embed_dim (int): Common embedding dimension.
        n_classes (int): Number of output classes.
    """
    def __init__(
        self,
        encoder_channels,
        embed_dim=256,
        n_classes=4
    ):
        super().__init__()

        self.mlps = nn.ModuleList([
            MLP(channel, embed_dim)
            for channel in encoder_channels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(
                embed_dim * len(encoder_channels),
                embed_dim,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(
            embed_dim,
            n_classes,
            kernel_size=1
        )

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): Encoder feature maps.

        Returns:
            Tensor: Per-pixel class scores.
        """
        target_size = features[0].shape[2:]
        projected = []

        for feat, mlp in zip(features, self.mlps):
            x = mlp(feat)

            x = F.interpolate(
                x,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )

            projected.append(x)

        x = torch.cat(projected, dim=1)
        x = self.fuse(x)

        return self.classifier(x)


class SegFormer(nn.Module):
    """
    SegFormer architecture using a MixVisionTransformer encoder and
    a lightweight All-MLP decoder head. The model is trained from
    scratch and produces pixel-level class scores for land-cover
    semantic segmentation.

    Args:
        encoder_name (str): MiT encoder variant.
        in_channels (int): Number of input channels.
        n_classes (int): Number of output classes.
    """
    def __init__(
        self,
        encoder_name="mit_b2",
        in_channels=6,
        n_classes=4
    ):
        super().__init__()

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            weights=None
        )

        stage_channels = self.encoder.out_channels[2:]

        self.head = SegFormerHead(
            stage_channels,
            embed_dim=256,
            n_classes=n_classes
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Input image batch.

        Returns:
            Tensor: Per-pixel class scores at the input resolution.
        """
        input_size = x.shape[2:]

        features = self.encoder(x)[2:]
        out = self.head(features)

        return F.interpolate(
            out,
            size=input_size,
            mode='bilinear',
            align_corners=False
        )