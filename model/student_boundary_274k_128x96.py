import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# DEPTHWISE-SEPARABLE CONVOLUTION
# ============================================================

class DSConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.dw_bn = nn.BatchNorm2d(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )

        self.pw_bn = nn.BatchNorm2d(out_channels)

        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.dw_bn(x)
        x = self.act(x)

        x = self.pointwise(x)
        x = self.pw_bn(x)
        x = self.act(x)

        return x


# ============================================================
# INVERTED RESIDUAL BLOCK
# ============================================================

class InvertedResidual(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expansion=2
    ):
        super().__init__()

        hidden = in_channels * expansion

        self.use_residual = (
            stride == 1
            and in_channels == out_channels
        )

        self.expand = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True)
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden,
                bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                hidden,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        x = self.expand(x)
        x = self.depthwise(x)
        x = self.project(x)

        if self.use_residual:
            x = x + identity

        return x


# ============================================================
# ENCODER
# Same encoder as the proven 269K baseline.
# ============================================================

class BoundaryStudentEncoder128x96(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = nn.Sequential(
            InvertedResidual(32, 48, stride=2, expansion=2),
            InvertedResidual(48, 48, stride=1, expansion=2)
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(48, 64, stride=2, expansion=2),
            InvertedResidual(64, 64, stride=1, expansion=2)
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(64, 96, stride=2, expansion=2),
            InvertedResidual(96, 96, stride=1, expansion=2)
        )

        self.stage4 = nn.Sequential(
            InvertedResidual(96, 128, stride=1, expansion=2),
            InvertedResidual(128, 128, stride=1, expansion=2)
        )

    def forward(self, x):
        features = []

        # 48x64, high-resolution structural feature
        x = self.stem(x)
        features.append(x)

        # 24x32
        x = self.stage1(x)
        features.append(x)

        # 12x16
        x = self.stage2(x)
        features.append(x)

        # 6x8
        x = self.stage3(x)
        features.append(x)

        # 6x8
        x = self.stage4(x)
        features.append(x)

        return features


# ============================================================
# BOUNDARY-PRESERVING DECODER
#
# Difference from 269K baseline:
# The final 48x64 decoder feature is concatenated with x0,
# the original 32-channel 48x64 high-resolution encoder feature.
#
# A 1x1 64->32 projection keeps the deployed model tiny and
# preserves compatibility with the original 32->16 refinement.
# ============================================================

class BoundaryStudentDecoder128x96(nn.Module):

    def __init__(self):
        super().__init__()

        self.dec4 = DSConv(128, 96)

        self.dec3 = DSConv(192, 64)

        self.dec2 = DSConv(128, 48)

        self.dec1 = DSConv(96, 32)

        # New boundary fusion layer:
        # [decoder 32 + encoder x0 32] -> 32
        self.boundary_fuse = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # Keep the original refinement and output layers unchanged.
        self.refine = nn.Sequential(
            nn.Conv2d(
                32,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU6(inplace=True)
        )

        self.output = nn.Conv2d(
            16,
            1,
            kernel_size=1
        )

    def forward(self, features):

        x0, x1, x2, x3, x4 = features

        # 6x8
        x = self.dec4(x4)

        # 6x8 -> 12x16
        x = F.interpolate(
            x,
            size=x3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)

        # 12x16 -> 24x32
        x = F.interpolate(
            x,
            size=x2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)

        # 24x32 -> 48x64
        x = F.interpolate(
            x,
            size=x1.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)

        # 24x32 -> 48x64
        x = F.interpolate(
            x,
            size=x0.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # NEW: preserve early high-resolution structure.
        # x0 is 32x48x64 and x is now 32x48x64.
        x = torch.cat([x, x0], dim=1)
        x = self.boundary_fuse(x)

        # 48x64 -> 96x128
        x = F.interpolate(
            x,
            size=(96, 128),
            mode="bilinear",
            align_corners=False
        )

        x = self.refine(x)
        x = self.output(x)

        # Normalized inverse depth
        x = torch.sigmoid(x)

        return x


# ============================================================
# COMPLETE MODEL
# ============================================================

class DepthStudentBoundary274K128x96(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = BoundaryStudentEncoder128x96()
        self.decoder = BoundaryStudentDecoder128x96()

    def forward(self, x):
        features = self.encoder(x)
        depth = self.decoder(features)
        return depth


# ============================================================
# BASELINE-COMPATIBLE INITIALIZATION
# ============================================================

def initialize_boundary_fusion_as_identity(model):
    """
    Initializes the new 64->32 fusion layer so the first 32
    input channels (the old decoder feature) pass through
    unchanged and the x0 channels initially contribute zero.

    This makes the new model's initial forward behavior very
    close to the original 269K model while allowing training
    to learn the high-resolution skip contribution.
    """

    conv = model.decoder.boundary_fuse[0]
    bn = model.decoder.boundary_fuse[1]

    with torch.no_grad():
        conv.weight.zero_()

        # Output channel i copies old decoder channel i.
        for i in range(32):
            conv.weight[i, i, 0, 0] = 1.0

        # Identity BatchNorm.
        bn.weight.fill_(1.0)
        bn.bias.zero_()
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    model = DepthStudentBoundary274K128x96()

    initialize_boundary_fusion_as_identity(model)

    x = torch.randn(1, 3, 96, 128)

    with torch.no_grad():
        y = model(x)

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("=" * 70)
    print("BOUNDARY-PRESERVING STUDENT — 128x96")
    print("=" * 70)

    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))

    print("Parameters:", f"{parameters:,}")
    print(
        "FP32 size:",
        f"{parameters * 4 / 1024 / 1024:.3f} MB"
    )
    print(
        "INT8 weight size:",
        f"{parameters / 1024 / 1024:.3f} MB"
    )

    print("\nFeature shapes:")

    features = model.encoder(x)

    for i, feature in enumerate(features):
        print(
            f"  Feature {i}:",
            tuple(feature.shape)
        )

    expected = 271617

    assert parameters == expected, (
        f"Unexpected parameter count: {parameters}; "
        f"expected {expected}"
    )

    assert y.shape == (
        1, 1, 96, 128
    ), f"Unexpected output shape: {y.shape}"

    print("\n" + "=" * 70)
    print("MODEL TEST PASSED")
    print("=" * 70)
