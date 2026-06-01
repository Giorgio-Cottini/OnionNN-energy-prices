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

import csv
import time
import math
import torch
import optuna
import numpy as np
import pandas as pd
from OnionNN import OnionNN 
from Embedders import TimestampEmbedder, BZNEmbedder
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import LambdaLR


#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
# Auxiliary classes

# Early stopping class
class EarlyStopping:
    def __init__(self, patience = 6, delta=0.001):
        self.patience   = patience
        self.delta      = delta
        self.min_loss   = float('inf')
        self.counter    = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.min_loss - self.delta:
            self.min_loss = val_loss
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#
# Learning rate warmup -> cosine is the most stable and widely used for large scale models (start, end = 1%)

def cosine_warmup_decay(step, total_steps, warmup_steps, min_lr_factor=1e-3):
    if step < warmup_steps:
        return (1 - min_lr_factor) * step / float(max(1, warmup_steps)) + min_lr_factor
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_factor + (1 - min_lr_factor) * cosine_decay

#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
# Training loop

def training_loop(model, optimizer, loss_function, train_dataloader, val_loader, epochs, device: str = 'cuda'):
    # Track epochs
    count = 1
    # Training loss history
    train_history = []
    val_history   = []
    
    #———— Early stopping
    early_stopping = EarlyStopping(patience = 6, delta = 0.0001)
    
    #———— Grad scaler -> this is the best feature i have ever discovered
    use_amp = device == 'cuda'
    scaler  = GradScaler(enabled = use_amp)

    #———— Set up learning rate spool up + decay
    total_steps  = len(train_dataloader) * epochs
    warmup_steps = int(0.05 * total_steps)
    lambda_optim = lambda s: cosine_warmup_decay(s, total_steps, warmup_steps,
                                           min_lr_factor=1e-2)
    #replicate the same schedule for every param group
    scheduler = LambdaLR(optimizer,
                       lr_lambda=[lambda_optim] * len(optimizer.param_groups))
    
    for epoch in range(epochs):
        # Initialize training specifics
        start_time  = time.time()
        train_loss  = 0.0
        num_batches = 0
        # Model in training mode
        model.train()
        # Iterate over batches in training dataloader
        for input_seq, zone_id, hour_id, dow_id, month_id, gas_val, onion_probs, target_seq in train_dataloader:

            # Move to device and put time dimension first (T, B, F)
            input_seq   = input_seq.transpose(0, 1).to(device, non_blocking = True)   # (T,B,F)
            zone_id     = zone_id.transpose(0, 1).to(device,   non_blocking = True)   # (T,B)
            hour_id     = hour_id.transpose(0, 1).to(device,   non_blocking = True)   # (T,B)
            dow_id      = dow_id.transpose(0, 1).to(device,    non_blocking = True)   # (T,B)
            month_id    = month_id.transpose(0, 1).to(device,  non_blocking = True)   # (T,B)
            gas_val     = gas_val.to(device,    non_blocking = True)                  # (B, 1) – no transpose
            onion_probs = onion_probs.to(device, non_blocking = True)                # (B, 24, 4)
            target_seq  = target_seq.to(device, non_blocking = True)                  # (B,24)

            #———— Forward Pass 
            with autocast(enabled = use_amp, dtype = torch.float16): # -> crazy speed up
                # Forward step
                predictions = model(input_seq, zone_id, hour_id, dow_id, month_id, gas_val, onion_probs)  # (B,24)
                # Loss function
                loss        = loss_function(predictions, target_seq, onion_probs)
            
            #———— Backward Pass (a bit awkward due to gradScaler)
            # Zero out gradients
            optimizer.zero_grad(set_to_none = True)
            # Backward step
            scaler.scale(loss).backward()
            # Gradient clipping -> kinda useful at the beginning, maybe improve clipping logic later on
            if not torch.isnan(loss):
                scaler.step(optimizer)  # Update parameters
                scaler.update()         # Update the scale for next iteration
                scheduler.step()        # Update learning rate
            
            # Update loss and batches counter
            train_loss  += loss.item()
            num_batches += 1

        # Average loss
        avg_loss = train_loss / num_batches
        # Append value to losses
        train_history.append(avg_loss)
        # Validation loop
        val_loss = validation_loop(model, loss_function, val_loader, device)
        val_history.append(val_loss)

        print(f"Epoch: {count:>2}  |  Train: {avg_loss:.5f}  |  Val: {val_loss:.5f}  |  Time: {time.time() - start_time :.2f}s")
        count += 1

        # Stop if avg loss is increasing
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Stopping early at epoch {epoch+1}")
            break

    return train_history, val_history


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#
# Validation loop
@torch.no_grad()
def validation_loop(model, loss_function, dataloader, device: str = 'cuda'):
    # Model in evaluation mode
    model.eval()
    # Initialize validation specifics
    validation_loss = 0.0
    num_batches     = 0

    for input_seq, zone_id, hour_id, dow_id, month_id, gas_val, onion_probs, target_seq in dataloader:
        # Move to device and put time dimension first (T, B, F)
        input_seq  = input_seq.transpose(0, 1).to(device, non_blocking = True)
        zone_id    = zone_id.transpose(0, 1).to(device,   non_blocking = True)
        hour_id    = hour_id.transpose(0, 1).to(device,   non_blocking = True)
        dow_id     = dow_id.transpose(0, 1).to(device,    non_blocking = True)
        month_id   = month_id.transpose(0, 1).to(device,  non_blocking = True)
        gas_val    = gas_val.to(device,    non_blocking   = True)                  # (B, 1) – no transpose
        onion_probs = onion_probs.to(device, non_blocking = True)                # (B, 24, 4)
        target_seq = target_seq.to(device, non_blocking   = True)          # (B, 24)

        #———— Forward Pass (no grad)
        predictions      = model(input_seq, zone_id, hour_id, dow_id, month_id, gas_val, onion_probs)  # (B, 24)
        validation_loss += loss_function(predictions, target_seq, onion_probs).item()
        num_batches     += 1

    return validation_loss / num_batches

#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#
# Dataset Tensorization

class CustomTensorDataset(Dataset):
    """
    Tensor-ready sliding-window dataset.

    Outputs per sample
    ------------------
    x_seq : (seq_len, F)   float32  - all numerical features
    zone  : (seq_len,)     int64    - BZN id repeated over the window
    hr    : (seq_len,)     int64    - hour index   0-23
    dow   : (seq_len,)     int64    - day-of-week  0-6
    mon   : (seq_len,)     int64    - month index 0-11
    y_seq : (pred_len,)    float32  - target price (univariate)
    """

    def __init__(
        self,
        scaled_data_dict: dict[str, np.ndarray],   # {zone: (T, F)}
        hours     : np.ndarray,                  # (T,)
        dows      : np.ndarray,
        months    : np.ndarray,
        seq_len   : int,
        pred_len  : int,
        price_idx : int,
        gas_idx   : int        
    ):
        
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.price_idx = price_idx
        self.gas_idx   = gas_idx

        # map BZN name → consecutive int for Embedding
        self.bzn_to_idx = {name: idx for idx, name in enumerate(scaled_data_dict.keys())}

        # build the index once- assumes all zones share the same T
        rng_limit = len(hours) - seq_len - pred_len + 1
        self.data: list[tuple] = []

        for zone_name, series in scaled_data_dict.items():
            z_id = self.bzn_to_idx[zone_name]

            # sliding window over the full horizon
            for i in range(0, rng_limit, 24): # keep or leave 24 just based on results
                x_slice    = series[i : i + seq_len, :-1]    # keep all columns except last (gas)
                gas_scalar = series[i + seq_len - 1, -1]     # gas is always the last column

                y_slice    = series[i + seq_len : i + seq_len + pred_len, self.price_idx]
                gas_scalar = series[i + seq_len - 1, -1]  # gas is last column

                self.data.append((
                    x_slice,
                    hours[i  : i + seq_len],
                    dows[i   : i + seq_len],
                    months[i : i + seq_len],
                    y_slice,
                    gas_scalar,
                    z_id
                ))

    # ————————————————————————————————————————————————————————————————
    def __len__(self) -> int:
        return len(self.data)

    # ————————————————————————————————————————————————————————————————
    def __getitem__(self, idx: int):
        x, hr, dw, mo, y, gas, zone = self.data[idx]

        # to-tensor with pinned memory for faster GPU transfer
        x_t   = torch.as_tensor(x,  dtype=torch.float32)         # (T, F)
        hr_t  = torch.as_tensor(hr, dtype=torch.long)
        dw_t  = torch.as_tensor(dw, dtype=torch.long)
        mo_t  = torch.as_tensor(mo, dtype=torch.long)
        gas_t = torch.tensor([gas], dtype=torch.float32)        # (1,)  ← NEW
        y_t   = torch.as_tensor(y,  dtype=torch.float32)        # (pred_len,)
        z_t   = torch.full((self.seq_len,), zone, dtype=torch.long)

        return x_t, z_t, hr_t, dw_t, mo_t, gas_t, y_t


class OnionTensorDataset(CustomTensorDataset):
    """
    Extends CustomTensorDataset by including onion model outputs (onion probabilities).
    
    Outputs per sample
    ------------------
    x_seq        : (seq_len, F)   float32  - all numerical features
    zone         : (seq_len,)     int64    - BZN id repeated over the window
    hr           : (seq_len,)     int64    - hour index   0-23
    dow          : (seq_len,)     int64    - day-of-week  0-6
    mon          : (seq_len,)     int64    - month index 0-11
    gas_scalar   : (1,)           float32  - last gas price of the input window
    onion_vec  : (24, 4)        float32  - onion probabilities at forecast start
    y_seq        : (pred_len,)    float32  - target price
    """

    def __init__(
        self,
        scaled_data_dict    : dict[str, np.ndarray],
        onion_probs_dict    : dict[str, np.ndarray],  # {zone: (T, 24, 4)}
        hours               : np.ndarray,
        dows                : np.ndarray,
        months              : np.ndarray,
        seq_len             : int,
        pred_len            : int,
        price_idx           : int = 0,
        gas_idx             : int = 0
    ):
        super().__init__(scaled_data_dict, hours, dows, months, seq_len, pred_len, price_idx, gas_idx)
        self.onion_probs_dict = onion_probs_dict

        # Overwrite self.data to include onion vectors
        self.data = []
        rng_limit = len(hours) - seq_len - pred_len + 1

        for zone_name, series in scaled_data_dict.items():
            z_id = self.bzn_to_idx[zone_name]
            onion_array = onion_probs_dict[zone_name]

            for i in range(0, rng_limit, 24):
                x_slice     = series[i : i + seq_len, :-1]
                gas_scalar  = series[i + seq_len - 1, -1]
                y_slice     = series[i + seq_len : i + seq_len + pred_len, self.price_idx]
                onion_vec = onion_array[i + seq_len - 1]  # (24, 4)

                self.data.append((
                    x_slice,
                    hours[i  : i + seq_len],
                    dows[i   : i + seq_len],
                    months[i : i + seq_len],
                    y_slice,
                    gas_scalar,
                    onion_vec,
                    z_id
                ))

    def __getitem__(self, idx: int):
        x, hr, dw, mo, y, gas, onion, zone = self.data[idx]

        x_t   = torch.as_tensor(x,       dtype=torch.float32)   # (T, F)
        hr_t  = torch.as_tensor(hr,      dtype=torch.long)
        dw_t  = torch.as_tensor(dw,      dtype=torch.long)
        mo_t  = torch.as_tensor(mo,      dtype=torch.long)
        y_t   = torch.as_tensor(y,       dtype=torch.float32)   # (24,)
        gas_t = torch.tensor([gas],      dtype=torch.float32)   # (1,)
        onion_t   = torch.as_tensor(onion, dtype=torch.float32)   # (24, 4)
        z_t   = torch.full((self.seq_len,), zone, dtype=torch.long)

        return x_t, z_t, hr_t, dw_t, mo_t, gas_t, onion_t, y_t

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#

def preprocess_and_scale(parquet_file: str):
    """Preprocesses the hourly parquet and returns scaled tensors per zone.

    The **gas_price** column is:
        1. Standard-scaled *globally* so every zone sees the same latent series.
        2. Re-attached (without further scaling) to each zone-specific feature matrix
           so downstream code can slice it with ``gas_idx``.

    Returns
    -------
    processed : dict[str, np.ndarray]
        ``{zone → (T, F)}`` where the last column is the globally-scaled gas price.
    scalers   : dict[str, StandardScaler]
        Per-zone scalers fitted on *local* features only (gas excluded).
    zones     : list[str]
    hours, dows, months : np.ndarray, shape (T,)
        Calendar indices.
    feature_names : list[str]
        Names of the *local* features (gas excluded).
    gas_idx  : int
        Absolute index of the gas column inside every ``processed[zone]`` array.
    """
    # ── read parquet ──────────────────────────────────────────────────────
    df    = pd.read_parquet(parquet_file)
    zones = sorted(df["zone"].unique())

    # ── 1. global standardisation of the latent gas feature ──────────────
    gas_scaler       = StandardScaler().fit(df[["gas_price"]])
    df["gas_price"] = gas_scaler.transform(df[["gas_price"]]).astype("float32")

    # ── 2. prepare structures ────────────────────────────────────────────
    processed, scalers = {}, {}

    ts_all     = df["timestamp"].drop_duplicates().sort_values()
    full_index = pd.date_range(start=ts_all.min(), end=ts_all.max(), freq="h")

    all_features  = [c for c in df.columns if c not in ("timestamp", "zone")]
    feature_names = [c for c in all_features if c != "gas_price"]  # local features only
    gas_idx       = len(feature_names)                               # will be appended last

    # ── 3. per-zone processing ───────────────────────────────────────────
    for zone in zones:
        sub_df = df[df["zone"] == zone].copy()
        sub_df = (
            sub_df.set_index("timestamp")
                  .sort_index()
                  .loc[~sub_df.index.duplicated(keep="first")]
                  .reindex(full_index)
        )

        # interpolate & fill missing on *all* columns to keep alignment
        sub_df[all_features] = (
            sub_df[all_features]
                .interpolate(method="linear", limit_direction="both")
                .fillna(0)
        )

        # scale local features
        scaler       = StandardScaler()
        scaled_local = scaler.fit_transform(sub_df[feature_names].astype("float32"))

        # append the already-scaled gas column (shape (T,1))
        gas_col    = sub_df["gas_price"].values.astype("float32").reshape(-1, 1)
        scaled     = np.concatenate([scaled_local, gas_col], axis=1)

        # sanity check
        if np.isnan(scaled).any():
            raise ValueError(f"NaNs in scaled data for zone {zone}")

        processed[zone] = scaled
        scalers[zone]   = scaler  # gas scaler not returned; you said it's not needed

    # ── 4. calendar indices (shared across zones) ────────────────────────
    hours  = full_index.hour.astype("int16").to_numpy()
    dows   = full_index.dayofweek.astype("int16").to_numpy()
    months = (full_index.month - 1).astype("int16").to_numpy()

    return processed, scalers, zones, hours, dows, months, feature_names, gas_idx, full_index

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#

def extract_onion_tensor(df: pd.DataFrame, zones: list[str]) -> dict[str, np.ndarray]:
    onion_vars = ["p_norm", "p_tail", "p_neg", "p_spike"]
    onion_tensor_dict = {}

    for z in zones:
        sub_df = df[df["zone"] == z].sort_values(["timestamp", "horizon"])

        # Get unique timestamps and ensure all 24 horizons are there for each
        timestamps = sub_df["timestamp"].drop_duplicates().sort_values()
        T = len(timestamps)

        # Reshape to (T, 24, 4)
        block = np.zeros((T, 24, 4), dtype="float32")

        for j, var in enumerate(onion_vars):
            pivoted = sub_df.pivot(index="timestamp", columns="horizon", values=var).sort_index()
            # Ensure correct horizon order (1–24)
            block[:, :, j] = pivoted[[h for h in range(1, 25)]].to_numpy()

        onion_tensor_dict[z] = block

    return onion_tensor_dict


#——————————————————————————————————————————————————————————————————————————————————————————————————————————————#


#:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:<=>:<->:#