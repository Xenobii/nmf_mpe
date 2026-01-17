import torch
import torch.nn as nn
from nnAudio.features import CQT2010v2



class CQT(nn.Module):
    def __init__(
        self, 
        sr: int,
        hop_length: int,
        fmin: float,
        n_bins: int,
        bins_per_semitone: int
    ):
        super().__init__()
        
        self.cqt = CQT2010v2(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=12 * bins_per_semitone,
            verbose=False,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        spec = self.cqt(wav)
        
        if spec.shape[-1] > 256:
            spec = spec[..., :256]

        spec = torch.log1p(torch.abs(spec))
        return spec