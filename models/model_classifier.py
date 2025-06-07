# models/model_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
#                            ResNet-12 Implementation                           #
#                                                                              #
# This file replaces the old AudioMLP. It defines:                             #
#   1. conv3x3: helper for 3×3 convolutions with padding                       #
#   2. BasicBlock: two 3×3 Conv → BatchNorm → ReLU layers, plus optional        #
#      downsampling via a 1×1 conv if needed.                                   #
#   3. AudioResNet12: four sequential “stages,” each consisting of BasicBlock  #
#      modules and a 2×2 MaxPool to downsample.                                 #
#   4. A final classifier head that global-pools and outputs `num_classes`     #
#                                                                              #
# Input shape: (batch_size, 3, n_mels, time_frames)                             #
# Output shape: (batch_size, num_classes)                                      #
################################################################################


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """
    3×3 convolution with padding=1, bias=False.
    We default to bias=False because we always follow it with BatchNorm2d.
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    A “basic” residual block:
      (1) conv3x3 → BatchNorm2d → ReLU
      (2) conv3x3 → BatchNorm2d
      (3) If in_channels != out_channels or stride != 1, then downsample the
          residual connection via a 1×1 conv with the same stride.
      (4) Add the residual (possibly downsampled), then apply ReLU.
    """

    expansion: int = 1  # For BasicBlock, output channels = out_channels × expansion (expansion=1)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super(BasicBlock, self).__init__()

        # First 3×3 conv
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3×3 conv (always stride=1 here)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the input→output dimensions differ (or stride != 1), we need a downsampling layer:
        self.downsample = None
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            # 1×1 conv to match #channels and spatial size
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for one BasicBlock. Computes:
           out = Conv3x3 → BN → ReLU → Conv3x3 → BN
           residual = (possibly downsample x)
           out = out + residual
           out = ReLU(out)
        """
        identity = x  # save for the residual connection

        # First conv → BN → ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second conv → BN
        out = self.conv2(out)
        out = self.bn2(out)

        # If downsample is defined, apply to the identity (residual path)
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add the residual
        out += identity
        out = self.relu(out)
        return out


class AudioResNet12(nn.Module):
    """
    A ResNet-12-style CNN for spectogram inputs with dynamic channels:
      - Four “stages,” each stage = BasicBlock × N, followed by 2×2 MaxPool.
      - Channel progression:      1 → 64 → 128 → 256 → 512
      - After the last stage, we do a global average pool over the remaining
        (freq × time) dimensions, then a linear layer to `num_classes`.
    """

    def __init__(
        self,
        n_mels: int,
        time_frames: int,
        num_classes: int,
        blocks_per_stage: int = 1,
    ):
        """
        Args:
          n_mels      : number of Mel-frequency bins (height of input spectrogram)
          time_frames : number of time frames (width of input spectrogram)
          num_classes : number of target classes (50 for ESC-50)
          blocks_per_stage: how many BasicBlocks in each of the four stages
        """
        super(AudioResNet12, self).__init__()

        # We assume input has shape (B, 3, n_mels, time_frames).

        # ----------------------------------------------------------------------
        # (1) Initial convolution: expand 1 channel → 64 channels, keep spatial dims
        # ----------------------------------------------------------------------
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # No maxpool here; we will downsample in each stage explicitly.

        # ----------------------------------------------------------------------
        # (2) Four “stages” of BasicBlocks. Each stage does:
        #       BasicBlock × blocks_per_stage  →  2×2 MaxPool to half spatial size
        # ----------------------------------------------------------------------
        self.layer1 = self._make_stage(
            out_channels=64, blocks=blocks_per_stage, stride=1
        )  # output: 64 × (n_mels) × (time_frames)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # after pool1: 64 × (n_mels/2) × (time_frames/2)

        self.layer2 = self._make_stage(
            out_channels=128, blocks=blocks_per_stage, stride=1
        )  # output: 128 × (n_mels/2) × (time_frames/2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # after pool2: 128 × (n_mels/4) × (time_frames/4)

        self.layer3 = self._make_stage(
            out_channels=256, blocks=blocks_per_stage, stride=1
        )  # output: 256 × (n_mels/4) × (time_frames/4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # after pool3: 256 × (n_mels/8) × (time_frames/8)

        self.layer4 = self._make_stage(
            out_channels=512, blocks=blocks_per_stage, stride=1
        )  # output: 512 × (n_mels/8) × (time_frames/8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # after pool4: 512 × (n_mels/16) × (time_frames/16)

        # ----------------------------------------------------------------------
        # (3) Final classification head:
        #     - Global average pool over whatever remains (freq × time)
        #     - Flatten → Linear(num_features=512, num_classes)
        # ----------------------------------------------------------------------
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # ----------------------------------------------------------------------
        # (4) Initialize weights (following Kaiming initialization for Convs)
        # ----------------------------------------------------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        """
        Builds one ‘stage’ of the network, consisting of `blocks` BasicBlock modules.
        The first block may downsample if stride != 1 or in_channels != out_channels.
        All subsequent blocks in this stage have stride=1 and in_channels = out_channels.
        """
        layers = []
        # First block in this stage: might need downsampling if (in_channels != out_channels) or stride != 1.
        layers.append(
            BasicBlock(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        )
        # After that, in_channels for the next block equals out_channels * expansion (here expansion=1).
        self.in_channels = out_channels * BasicBlock.expansion

        # The remaining (blocks-1) blocks in this stage simply keep the same #channels, stride=1
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=1,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (B, 3, n_mels, time_frames)
        Output: (B, num_classes)
        """
        # ----------------------------------------------------------------------
        # (1) Initial conv → BN → ReLU
        # ----------------------------------------------------------------------
        out = self.conv1(x)   # shape: [B, 64, n_mels, time_frames]
        out = self.bn1(out)
        out = self.relu(out)

        # ----------------------------------------------------------------------
        # (2) Stage 1 → MaxPool
        # ----------------------------------------------------------------------
        out = self.layer1(out)  # still [B, 64, n_mels, time_frames]
        out = self.pool1(out)   # now [B, 64, n_mels/2, time_frames/2]

        # ----------------------------------------------------------------------
        # (3) Stage 2 → MaxPool
        # ----------------------------------------------------------------------
        out = self.layer2(out)  # [B, 128, n_mels/2, time_frames/2]
        out = self.pool2(out)   # [B, 128, n_mels/4, time_frames/4]

        # ----------------------------------------------------------------------
        # (4) Stage 3 → MaxPool
        # ----------------------------------------------------------------------
        out = self.layer3(out)  # [B, 256, n_mels/4, time_frames/4]
        out = self.pool3(out)   # [B, 256, n_mels/8, time_frames/8]

        # ----------------------------------------------------------------------
        # (5) Stage 4 → MaxPool
        # ----------------------------------------------------------------------
        out = self.layer4(out)  # [B, 512, n_mels/8, time_frames/8]
        out = self.pool4(out)   # [B, 512, n_mels/16, time_frames/16]

        # ----------------------------------------------------------------------
        # (6) Global average pool → flatten → classifier
        # ----------------------------------------------------------------------
        out = F.dropout2d(out, p=0.2, training=self.training)
        out = self.global_avg_pool(out)  # [B, 512, 1, 1]
        out = torch.flatten(out, 1)       # [B, 512]
        out = self.classifier(out)        # [B, num_classes]

        return out
