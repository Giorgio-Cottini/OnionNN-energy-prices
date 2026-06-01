#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
#                                                                                                             #
# Machine Learning for Finance                                                ########   ########             #
#                                                                            ##         ##                    #
# Training your LSTM has never been this easy                                ##   ####  ##                    #
#                                                                            ##     ##  ##                    #
# Date of creation: 21/05/2025                                                ########   ########             #
#                                                                                                             #
#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#

from typing import Optional
import Embedders
import torch
import torch.nn as nn
from torch import Tensor


#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
# Deep LSTM

class OnionNN(nn.Module):
    def __init__(self,
                 *,
                 input_size    : int,      # numerical features (prices + extra vars)
                 hidden_size   : int,
                 dense_dim     : int = 128,
                 output_size   : int   = 24,
                 time_embedder : Embedders.TimestampEmbedder,
                 bzn_embedder  : Embedders.BZNEmbedder,
                 num_layers    : int   = 2,
                 dropout       : float = 0.2,
                 device        : str   = "cuda"):

        super().__init__()

        #————— Embedders sanity check
        if time_embedder is None or bzn_embedder is None:
            raise ValueError("Embedders missing during model initialization.")
        self.time_embedder = time_embedder
        self.bzn_embedder  = bzn_embedder
        # Get embedding specifics
        bzn_dim  = bzn_embedder.embedding_dim
        d_hour   = time_embedder.hour_emb.embedding_dim
        d_day    = time_embedder.day_emb.embedding_dim
        d_month  = time_embedder.month_emb.embedding_dim
        feat_dim = input_size + bzn_dim + d_hour + d_day + d_month

        #————— Core settings
        self.device      = device
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        
        #————— LSTM stack
        # feature dimension = input dimension + sum of embeddings' dimensions
        self.lstm = nn.LSTM(
            input_size  = feat_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = False,          # (T, B, F) -> i like it better
            device      = device,
        )

        #————— Fusion + Output head 
        #self.fusion = ConvOnionFusion()
        self.fusion = ConvOnionFusion()
        self.dense_out = nn.Sequential(
            nn.Linear(hidden_size + 1 + 192, dense_dim, device = device), # -> convolution
            #nn.Linear(hidden_size + 1 + 120, dense_dim, device = device),# -> simple
            nn.GELU(),
            nn.Linear(dense_dim, output_size, device = device)
        )     

    #——————————————————————————————————————————————————————————————————————————————————————————————————————————#
    # Trainability toggles
    def set_trainable(
        self,
        *,                          # keyword-only
        time_embed : bool,
        bzn_embed  : bool,
        lstm       : bool,
        dense      : bool
    ) -> None:
        """Switch gradients on/off for the main sub-modules."""
        # timestamp embeddings
        for p in self.time_embedder.parameters():
            p.requires_grad = time_embed
        # BZN embeddings
        for p in self.bzn_embedder.parameters():
            p.requires_grad = bzn_embed
        # LSTM stack
        for p in self.lstm.parameters():
            p.requires_grad = lstm
        # Dense output head (and any extra linear layers you might add)
        for p in self.dense_out.parameters():
            p.requires_grad = dense
       
    #——————————————————————————————————————————————————————————————————————————————————————————————————————————#
    # Forward pass
    def forward(
        self,
        price_seq    : Tensor,     # (T, B, input_size)
        zone_id      : Tensor,     # (T, B)
        hour_id      : Tensor,     # (T, B)
        dow_id       : Tensor,     # (T, B)
        month_id     : Tensor,     # (T, B)
        gas_price    : Tensor,     # (B, 1)
        onion_probs  : Tensor = None,
        hidden_state : Optional[tuple[Tensor, Tensor]] = None  # (num_layers, B, H)
    ) -> Tensor:
        """
        Forward pass through the model.
        
        Parameters
        ──────────
        price_seq : (T, B, input_size (F))
        zone_id   : (T, B)
        hour_id   : (T, B)
        dow_id    : (T, B)
        month_id  : (T, B)
        gas_price : (T, B)

        hidden_state : (num_layers, B, H)
        
        Returns
        ───────
        (B, output_size)
        """

        #──── Embed categorical time and zone information
        # Learned zone embeddings
        z_vec = self.bzn_embedder(zone_id)               # (T, B, bzn_dim)
        # Sinusoidal and/or trainable embeddings
        h_vec = self.time_embedder.hour_emb(hour_id)     # (T, B, d_hour)
        d_vec = self.time_embedder.day_emb(dow_id)       # (T, B, d_day)
        m_vec = self.time_embedder.month_emb(month_id)   # (T, B, d_month)

        #──── Concatenate all feature streams
        # LSTM input: numeric features + embeddings
        x = torch.cat((price_seq, z_vec, h_vec, d_vec, m_vec), dim = -1)  # (T, B, F_total)

        #──── Pass through LSTM
        lstm_out, hidden_state = self.lstm(x, hidden_state)  # lstm_out: (T, B, H)

        # Extract the last timestep
        lstm_last_step = lstm_out[-1] # (B, H)

        #──── 1D Convolution + ReLU + flatten
        x_onion = self.fusion(onion_probs)  # ← outputs 192-dim

        #──── Concatenate dense layer's input ....
        x_out = torch.cat([lstm_last_step, gas_price, x_onion], dim= - 1)

        #──── Dense Output layer -> linear + GeLU + linear
        out = self.dense_out(x_out)  # (B, output_size)

        return out

    #——————————————————————————————————————————————————————————————————————————————————————————————————————————#
    # Setter 
     
    @torch.no_grad()
    def set_bzn_weight(self, weight: Tensor) -> None:
        """
        Overwrite the learned BZN embedding with a user-supplied matrix.

        weight : (n_bzn, bzn_dim) tensor
        """
        target = self.bzn_embedder.embedding
        if weight.shape != target.weight.shape:
            raise ValueError(f"Shape mismatch: expected {target.weight.shape}, got {weight.shape}")
        target.weight.copy_(weight.to(self.device))
    #——————————————————————————————————————————————————————————————————————————————————————————————————————————#
    # Getter

    @torch.no_grad()
    def get_bzn_vectors(self, ids: Optional[torch.Tensor | list[int]] = None) -> torch.Tensor:
        """
        Return BZN embedding vectors.
        ids = None        -> full (n_bzn, dim) table
        ids = list/Tensor -> specific rows
        """
        if isinstance(ids, list):
            ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if ids is None:
            return self.bzn_embedder.weight.detach()          # (n_bzn, dim)
        return self.bzn_embedder(ids).detach()                # (k, dim)

    #——————————————————————————————————————————————————————————————————————————————————————————————————————————#
    # Timestamp getter

    @torch.no_grad()
    def get_timestamp_vectors(
        self,
        *, hour  : Optional[int] = None,
           dow   : Optional[int] = None,
           month : Optional[int] = None
    ):
        """
        Fetch individual timestamp vectors; leave arg None to skip.
        Returns a tuple (h_vec, d_vec, m_vec) with None where skipped.
        """
        he = self.time_embedder.hour_emb
        de = self.time_embedder.day_emb
        me = self.time_embedder.month_emb

        h_vec = he(torch.tensor([hour], device=self.device)).squeeze(0) if hour  is not None else None
        d_vec = de(torch.tensor([dow],  device=self.device)).squeeze(0) if dow   is not None else None
        m_vec = me(torch.tensor([month],device=self.device)).squeeze(0) if month is not None else None
        return h_vec, d_vec, m_vec

    #——————————————————————————————————————————————————————————————————————————————————————————————————————————#
    # Provisory
    
    def init_hidden(self, batch_size: int) -> tuple[Tensor, Tensor]:
        '''
        Create fresh (h_0, c_0) tensors on correct device for external usage,
        avoiding per-step allocations inside training loops.
        '''
        shape = (self.num_layers, batch_size, self.hidden_size)
        h0    = torch.zeros(shape, device = self.device)
        c0    = torch.zeros(shape, device = self.device)
        return h0, c0
    
#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
class ConvOnionFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=5, out_channels=8, kernel_size=3, padding=1)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.flat = nn.Flatten()
        self.norm = nn.LayerNorm(8 * 24)  # == 192

    def forward(self, onion_probs):  # onion_probs: (B, 24, 4)
        with torch.no_grad():
            max_val, max_idx = torch.max(onion_probs, dim=-1, keepdim=True)     # (B, 24, 1)
            onehot = torch.zeros_like(onion_probs).scatter_(-1, max_idx, 1.0)   # (B, 24, 4)

        x = torch.cat([onehot, max_val], dim=-1)  # (B, 24, 5)
        x = x.permute(0, 2, 1)                    # → (B, 5, 24)

        x = self.conv(x)                          # → (B, 8, 24)
        x = self.act(x)
        x = x.reshape(x.size(0), -1)              # → (B, 192)
        x = self.norm(x)
        x = self.drop(x)

        return x


class SimpleOnionFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, onion_probs):  # (B, 24, 4)
        with torch.no_grad():
            conf, max_idx = torch.max(onion_probs, dim=-1, keepdim=True)    # (B, 24, 1)
            onehot = torch.zeros_like(onion_probs).scatter_(-1, max_idx, 1.0)  # (B, 24, 4)

        x = torch.cat([onehot, conf], dim=-1)     # (B, 24, 5)
        return x.view(x.size(0), -1)              # (B, 120)

#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
