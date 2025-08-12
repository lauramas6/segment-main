# models/cnn/fcdensenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

class _DenseLayer(nn.Module):
    def __init__(self, in_ch, growth):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = _conv3x3(in_ch, growth)
    def forward(self, x):
        out = F.relu(self.bn(x), inplace=True)
        out = self.conv(out)
        return torch.cat([x, out], dim=1)

class _DenseBlock(nn.Module):
    def __init__(self, in_ch, n_layers, growth):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_layers):
            layers.append(_DenseLayer(ch, growth))
            ch += growth
        self.block = nn.Sequential(*layers)
        self.out_ch = ch
    def forward(self, x):
        return self.block(x)

class _TransitionDown(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x = F.relu(self.bn(x), inplace=True)
        x = self.conv(x)
        return self.pool(x)

class _TransitionUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
    def forward(self, x):
        return self.deconv(x)

class FCDenseNet103(nn.Module):
    """
    Minimal FC-DenseNet103 (Tiramisu) with default 103-layer config:
    down_blocks=[4,5,7,10,12], bottleneck=15, up_blocks=[12,10,7,5,4]
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=2,
        growth_rate=16,
        init_features=48,
        down_blocks=(4,5,7,10,12),
        up_blocks=(12,10,7,5,4),
        bottleneck_layers=15,
    ):
        super().__init__()
        self.stem = _conv3x3(in_channels, init_features)

        # Down path
        self.down_blocks = nn.ModuleList()
        self.td = nn.ModuleList()
        ch = init_features
        self.skip_channels = []
        for n in down_blocks:
            db = _DenseBlock(ch, n, growth_rate)
            self.down_blocks.append(db)
            ch = db.out_ch
            self.skip_channels.append(ch)
            self.td.append(_TransitionDown(ch))

        # Bottleneck
        self.bottleneck = _DenseBlock(ch, bottleneck_layers, growth_rate)
        ch = self.bottleneck.out_ch

        # Up path
        self.tu = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for n, sk in zip(up_blocks, reversed(self.skip_channels)):
            self.tu.append(_TransitionUp(ch, sk))
            ch = sk + sk  # after concat with skip of size sk
            ub = _DenseBlock(ch, n, growth_rate)
            self.up_blocks.append(ub)
            ch = ub.out_ch

        self.classifier = nn.Conv2d(ch, num_classes, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.stem(x)
        skips = []
        for db, td in zip(self.down_blocks, self.td):
            x = db(x)
            skips.append(x)
            x = td(x)

        x = self.bottleneck(x)

        for tu, ub, sk in zip(self.tu, self.up_blocks, reversed(skips)):
            x = tu(x)
            if x.shape[-2:] != sk.shape[-2:]:
                x = F.interpolate(x, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, sk], dim=1)
            x = ub(x)

        logits = self.classifier(x)
        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits

def get_fc_densenet103_model(CFG):
    return FCDenseNet103(
        in_channels=CFG.in_channels,
        num_classes=CFG.num_classes,
        growth_rate=getattr(CFG, "growth_rate", 16),
        init_features=getattr(CFG, "init_features", 48),
        down_blocks=getattr(CFG, "down_blocks", (4,5,7,10,12)),
        up_blocks=getattr(CFG, "up_blocks", (12,10,7,5,4)),
        bottleneck_layers=getattr(CFG, "bottleneck_layers", 15),
    )
