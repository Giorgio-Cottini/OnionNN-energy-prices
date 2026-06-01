from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
from torch import Tensor

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#
# Helper – deterministic sinusoidal table

def _sinusoidal_matrix(num_pos: int, dim: int, device: torch.device) -> Tensor:
    '''
    Standard Transformer-style sinusoidal table.
        PE[pos, 2i]   = sin( pos / 10000^(2i / dim) )
        PE[pos, 2i+1] = cos( pos / 10000^(2i / dim) )

    Args
    ----
    num_pos : number of discrete positions (e.g. 24 hours)
    dim     : embedding dimension -> must be even 
    device  : CUDA, MTS or CPU (don't do that)
    '''
    
    if dim % 2 != 0:
        raise ValueError("Sinusoidal dimension must be even (got {}).".format(dim))

    pos   = torch.arange(num_pos,   device = device, dtype=torch.float32).unsqueeze(1)       # (P, 1)
    two_i = torch.arange(0, dim, 2, device = device, dtype=torch.float32)                    # (dim/2,)
    div   = torch.exp(-math.log(10000.0) * two_i / dim)                                      # (dim/2,)

    pe = torch.zeros(num_pos, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)                                                     # even idx
    pe[:, 1::2] = torch.cos(pos * div)                                                     #  odd idx

    return pe

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#
# Timestamp embedder

class TimestampEmbedder(nn.Module):
    """Sinusoidal-initialised embeddings for hour, day-of-week and month."""

    def __init__(self,
                 dim_hour  : int = 8,
                 dim_day   : int = 4,
                 dim_month : int = 4,
                 *,
                 device  : str = "cuda"):
        
        super().__init__()
        device = torch.device(device)

        self.hour_emb  = nn.Embedding.from_pretrained(_sinusoidal_matrix(24, dim_hour, device))
        self.day_emb   = nn.Embedding.from_pretrained(_sinusoidal_matrix(7, dim_day, device))
        self.month_emb = nn.Embedding.from_pretrained(_sinusoidal_matrix(12, dim_month, device))

    def modules_tuple(self) -> Tuple[nn.Embedding, nn.Embedding, nn.Embedding]:
        return self.hour_emb, self.day_emb, self.month_emb 

    def get_weights(self):
        return self.hour_emb.weight.detach(), self.day_emb.weight.detach(), self.month_emb.weight.detach()

    def set_weights(self, hour_w, day_w, month_w):
        self.hour_emb.weight.data.copy_(hour_w)
        self.day_emb.weight.data.copy_(day_w)
        self.month_emb.weight.data.copy_(month_w)


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#
# BZN embedder

class BZNEmbedder(nn.Module):
    """Provides a (n_bzn, bzn_dim) embedding table, random or pretrained."""
    def __init__(self,
                 n_bzn      : int,
                 bzn_dim    : int,
                 *,
                 pretrained : Optional[Tensor] = None,
                 device     : str  = "cuda"):
        
        super().__init__()
        # Random if not provided
        if pretrained is None:
            self.embedding = nn.Embedding(n_bzn, bzn_dim, device = device)
        else:
            # Check dimensions
            if pretrained.shape != (n_bzn, bzn_dim):
                raise ValueError("Pretrained matrix shape mismatch.")
            # Embedding
            self.embedding = nn.Embedding.from_pretrained(pretrained.to(device))

    def get_module(self) -> nn.Embedding:
        return self.embedding

    def forward(self, ids: Tensor) -> Tensor:
        return self.embedding(ids)

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim

    @property
    def weight(self):
        return self.embedding.weight
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#