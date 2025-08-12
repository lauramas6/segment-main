# models/cnn/frrn_a.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

class _RU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(ch); self.bn2 = nn.BatchNorm2d(ch)
        self.conv1 = _conv3x3(ch, ch); self.conv2 = _conv3x3(ch, ch)
    def forward(self, x):
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        return x + out

class _FRRU(nn.Module):
    """
    Simple FRRU: merges full-res residual stream R with pooled stream P.
    Produces updated P (at P scale) and injects residual back to R.
    """
    def __init__(self, r_ch, p_ch, out_p_ch):
        super().__init__()
        self.merge = _conv3x3(r_ch + p_ch, out_p_ch)
        self.bn = nn.BatchNorm2d(out_p_ch)
        self.res_proj = nn.Conv2d(out_p_ch, r_ch, 1, bias=False)

    def forward(self, r, p):
        # Downsample R to P size if needed
        if r.shape[-2:] != p.shape[-2:]:
            r_ds = F.interpolate(r, size=p.shape[-2:], mode="bilinear", align_corners=False)
        else:
            r_ds = r
        x = torch.cat([r_ds, p], dim=1)
        p = F.relu(self.bn(self.merge(x)), inplace=True)
        # Inject to R at full-resolution
        inj = F.interpolate(self.res_proj(p), size=r.shape[-2:], mode="bilinear", align_corners=False)
        r = r + inj
        return r, p

class FRRN_A(nn.Module):
    """
    Minimal, trainable FRRN-A-style network (2 downs / 2 ups).
    Keeps the two-stream idea; lightweight for a quick baseline.
    """
    def __init__(self, in_channels=3, num_classes=2, base=48):
        super().__init__()
        self.stem = nn.Sequential(
            _conv3x3(in_channels, base), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            _RU(base), _RU(base)
        )
        self.pool = nn.MaxPool2d(2)

        # Down path (P stream gets pooled; R stays full-res)
        self.frru_d1 = _FRRU(r_ch=base,   p_ch=base,   out_p_ch=base*2)  # P@1/2
        self.frru_d2 = _FRRU(r_ch=base,   p_ch=base*2, out_p_ch=base*4)  # P@1/4

        # Bottleneck on P (lowest scale)
        self.bot = nn.Sequential(
            _conv3x3(base*4, base*4), nn.BatchNorm2d(base*4), nn.ReLU(inplace=True),
            _conv3x3(base*4, base*4), nn.BatchNorm2d(base*4), nn.ReLU(inplace=True),
        )

        # Up path (upsample P, fuse via FRRUs)
        self.up1 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2, bias=False)  # to 1/2
        self.frru_u1 = _FRRU(r_ch=base, p_ch=base*2, out_p_ch=base*2)
        self.up0 = nn.ConvTranspose2d(base*2, base,   2, stride=2, bias=False)  # to full
        self.frru_u0 = _FRRU(r_ch=base, p_ch=base,   out_p_ch=base)

        # Final refinement on R + head
        self.tail = nn.Sequential(_RU(base), _RU(base))
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        r = self.stem(x)
        p = self.pool(r)                   # init pooled stream from R (1/2)

        r, p = self.frru_d1(r, p)         # fuse @1/2
        p = self.pool(p)                   # 1/4
        r, p = self.frru_d2(r, p)         # fuse @1/4

        p = self.bot(p)

        p = self.up1(p)                    # 1/2
        r, p = self.frru_u1(r, p)
        p = self.up0(p)                    # full
        r, p = self.frru_u0(r, p)

        r = self.tail(r)
        out = self.head(r)
        if out.shape[-2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        return out

def get_frrn_a_model(CFG):
    return FRRN_A(in_channels=CFG.in_channels, num_classes=CFG.num_classes, base=getattr(CFG, "base", 48))
