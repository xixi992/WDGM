import torch
import torch.nn as nn

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    _HAS_PYTORCH_WAVELETS = True
except ImportError:
    DWTForward = None
    DWTInverse = None
    _HAS_PYTORCH_WAVELETS = False


class WaveletRandomMask(nn.Module):


    def __init__(self, wave: str = "haar", drop_rate: float = 0.3, separate_channels: bool = False):
        super().__init__()
        self.drop_rate = float(drop_rate)
        self.separate_channels = separate_channels

        if _HAS_PYTORCH_WAVELETS and self.drop_rate > 0:
            self.dwt = DWTForward(J=1, wave=wave)
            self.idwt = DWTInverse(wave=wave)
        else:
            self.dwt = None
            self.idwt = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if (not self.training) or self.drop_rate <= 0 or self.dwt is None or self.idwt is None:
            return x

        Yl, Yh_list = self.dwt(x)
        Yh = Yh_list[0]

        mask_lp = (torch.rand_like(Yl) > self.drop_rate).float()
        Yl = Yl * mask_lp

        if self.separate_channels:
            for i in range(Yh.shape[2]):
                mask_hp = (torch.rand_like(Yh[:, :, i]) > self.drop_rate).float()
                Yh[:, :, i] = Yh[:, :, i] * mask_hp
        else:
            mask_hp = (torch.rand_like(Yh) > self.drop_rate).float()
            Yh = Yh * mask_hp

        out = self.idwt((Yl, [Yh]))
        return out

