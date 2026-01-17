import torch 
import torch.nn as nn



def random_init(F: int, N: int) -> torch.Tensor:
    """
    Random initialization of W
    """
    W = torch.rand(F, N)
    return W / (W.max(dim=0, keepdim=True).values + 1e-8)


class NMFDict(nn.Module):
    """
    Only the dictionary W is trainable, activations H act as labels
    """
    def __init__(self, W: torch.Tensor):
        super().__init__()
        self.W = nn.Parameter(W)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.relu(self.W), H)


