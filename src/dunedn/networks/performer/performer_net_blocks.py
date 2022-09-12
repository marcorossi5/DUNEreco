from performer_pytorch import SelfAttention
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.attn = SelfAttention(dim)

    def forward(self, x):
        original_shape = x.shape
        batch_size = x.shape[0]
        output = self.attn(x.view((batch_size, -1, self.dim)))
        return output.view(original_shape)


class PreProcessBlock(nn.Module):
    def __init__(self, kernel_size, ic, oc):
        super(PreProcessBlock, self).__init__()
        ks = kernel_size
        kso2 = kernel_size // 2
        self.activ = nn.LeakyReLU(0.05)
        self.convs = nn.Sequential(
            nn.Conv2d(ic, oc, ks, padding=(kso2, kso2)),
            self.activ,
            nn.Conv2d(oc, oc, ks, padding=(kso2, kso2)),
            self.activ,
            nn.Conv2d(oc, oc, ks, padding=(kso2, kso2)),
            self.activ,
        )
        self.bn = nn.BatchNorm2d(oc)
        self.attn = AttentionLayer(oc)

    def forward(self, x):
        x = self.convs(x)
        x = self.activ(self.attn(x))
        return x


class HPF(nn.Module):
    """High Pass Filter"""

    def __init__(self, kernel_size, ic, oc):
        super().__init__()
        ks = kernel_size
        kso2 = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(ic, ic, 3, padding=1), nn.BatchNorm2d(ic), nn.LeakyReLU(0.05)
        )
        self.atns = nn.Sequential(
            AttentionLayer(ic),
            nn.Conv2d(ic, oc, ks, padding=(kso2, kso2)),
            AttentionLayer(oc),
        )
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.conv(x)
        x = self.atns(x)
        return x


class LPF(nn.Module):
    """Low Pass Filter"""

    def __init__(self, kernel_size, ic, oc):
        super().__init__()
        ks = kernel_size
        kso2 = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(ic, ic, 5, padding=2), nn.BatchNorm2d(ic), nn.LeakyReLU(0.05)
        )
        self.atns = nn.Sequential(
            AttentionLayer(ic),
            nn.BatchNorm2d(ic),
            nn.LeakyReLU(0.05),
            nn.Conv2d(ic, oc, ks, padding=(kso2, kso2)),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU(0.05),
            AttentionLayer(oc),
            nn.BatchNorm2d(oc),
            nn.LeakyReLU(0.05),
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.atns(y)
        return x + y


class PostProcessBlock(nn.Module):
    def __init__(self, kernel_size, ic, hc):
        super().__init__()
        ks = kernel_size
        kso2 = kernel_size // 2
        self.pipeline = nn.Sequential(
            nn.Conv2d(hc * 4, hc * 2, ks, padding=(kso2, kso2)),
            nn.BatchNorm2d(hc * 2),
            nn.LeakyReLU(0.05),
            nn.Conv2d(hc * 2, hc, ks, padding=(kso2, kso2)),
            nn.BatchNorm2d(hc),
            nn.LeakyReLU(0.05),
            nn.Conv2d(hc, ic, ks, padding=(kso2, kso2)),
        )

    def forward(self, x):
        return self.pipeline(x)
