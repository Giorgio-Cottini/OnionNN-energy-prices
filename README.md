# OnionNN — Day-Ahead Electricity Price Forecasting for European Bidding Zones

A deep-learning pipeline that forecasts the 24 hourly day-ahead electricity
prices of European bidding zones. The core model is a multi-layer LSTM that
combines numerical market features with learned categorical embeddings (zone,
hour, day-of-week, month) and a regime-probability signal produced by a
separate price-regime classifier (the _Onion_ component).

The project was developed as a _Machine Learning for Finance_ course project.

---

## At a Glance

|               |                                                                                  |
| ------------- | -------------------------------------------------------------------------------- |
| **Task**      | 24-hour day-ahead electricity price forecasting (EUR/MWh)                        |
| **Model**     | Multi-layer LSTM with learned zone/time embeddings and regime-probability fusion |
| **Data**      | 20 European bidding zones, hourly, 2019–2024 (ENTSO-E)                           |
| **Framework** | PyTorch                                                                          |
| **Context**   | Machine Learning for Finance — course project                                    |

---

## Motivation and Goals

Day-ahead electricity prices are volatile, strongly seasonal, and zone-specific.
Prices can go negative (renewable oversupply) or spike sharply (scarcity), and
these extreme regimes are precisely what simple regressors handle worst.

The goals of this project are:

- Build a single model that forecasts all 24 hourly prices of a day in one shot.
- Train one shared model across many bidding zones, using a learned **zone
  embedding** so a single network can specialize per market.
- Inject an explicit **price-regime signal** (normal / tail / negative / spike)
  into the forecaster so it is aware of likely extreme hours.
- Compare the resulting model against linear-regression and plain-LSTM baselines.

---

## Main Features

- **Multi-zone forecasting** over 20 European bidding zones (see list below),
  hourly resolution, 2019–2024.
- **Sequence-to-vector LSTM** (`OnionNN`) predicting a 24-hour price vector.
- **Learned embeddings**
  - `BZNEmbedder`: a trainable bidding-zone embedding table (optionally
    initialized from precomputed vectors).
  - `TimestampEmbedder`: sinusoidally-initialized embeddings for hour (24),
    day-of-week (7) and month (12).
- **Onion regime fusion** (`ConvOnionFusion`): a 1-D convolution over the
  per-hour regime probabilities `(p_norm, p_tail, p_neg, p_spike)`, fused into
  the dense output head alongside the LSTM state and the gas price.
- **Training utilities** (`OnionTrainer`): AMP mixed-precision training, cosine
  learning-rate warmup/decay, early stopping, per-submodule trainability
  toggles, and Optuna hyperparameter search.
- **Data pipeline** (`dataProcessor`): cleans raw ENTSO-E-style CSV exports,
  resamples to an hourly grid, merges per-zone feature files, attaches a global
  gas-price series, and writes a single compressed Parquet dataset.
- **Baselines** (`baselines/RNN.py`): a plain `DeepLSTM` without the Onion
  fusion, plus linear-regression and LSTM prediction dumps for comparison.

---

### Covered bidding zones

`CH, CZ, DE-LU, DK1, DK2, ES, FI, FR, IT-North, NL, NO1, NO2, NO3, NO4, NO5,
PT, SE1, SE2, SE3, SE4`

---

## Repository Structure

```
.
├── OnionScripts/                 # Model + training source
│   ├── OnionNN.py                # OnionNN model + ConvOnionFusion / SimpleOnionFusion
│   ├── Embedders.py              # BZNEmbedder, TimestampEmbedder
│   ├── OnionTrainer.py           # Datasets, training/validation loops, Optuna search
│   ├── dataProcessor.py          # Raw CSV cleaning, merging, Parquet export
│   ├── Onion-NN.ipynb            # End-to-end training/evaluation notebook
│   └── baselines/
│       └── RNN.py                # DeepLSTM baseline (no Onion fusion)
│
├── data/                         # Raw / intermediate inputs
│   ├── prices/  loads/  meteo/   # Per-zone market and weather series
│   ├── dayahead_wind_solar/      # Day-ahead wind & solar generation forecasts
│   ├── historical_forecasts/     # Weather forecast series + maps (per country)
│   └── raw_data/
│
├── merged_data/                  # Per-zone merged CSVs ({ZONE}_merged.csv)
├── onion_dataset.parquet         # Unified hourly dataset (all zones, 2019–2024)
├── processed_gas.csv             # Daily natural-gas price series
├── mds_embeddings.csv            # Precomputed zone embeddings (MDS)
│
├── trained_model/
│   ├── mse/                      # OnionNN_weights.pt + lr/lstm/onion prediction dumps
│   ├── forecasts/                # Per-zone forecast plots (PNG)
│   └── custom/                   # pred_vs_true_by_bzn.csv
│
├── pics/                         # Architecture and error-analysis figures
│
├── LSTM.ipynb                    # LSTM experiments (Colab)
├── embedding.ipynb               # Zone embeddings via MDS
├── dataset_processor.ipynb       # Dataset assembly
├── preliminary_analysis.ipynb    # Exploratory data analysis
└── output_analysis.ipynb         # Error / forecast analysis
```

---

## Technical Architecture

`OnionNN` is a sequence-to-vector model. A sliding window of past hourly prices
and features is encoded by a stacked LSTM; its final hidden state is fused with
a gas-price scalar and a convolutional summary of per-hour regime probabilities
before a small dense head emits the full 24-hour price vector. Zone identity and
calendar position are injected as learned embeddings rather than raw indices.

### Input dataset

`onion_dataset.parquet` holds one long table, hourly, all zones stacked:

| column            | description                                |
| ----------------- | ------------------------------------------ |
| `timestamp`       | hourly timestamp (2019-01-01 → 2024-12-31) |
| `zone`            | bidding-zone identifier                    |
| `Day-ahead Price` | target price (EUR/MWh)                     |
| `Actual Load`     | realized total load (MW)                   |
| `Solar`           | day-ahead solar generation forecast (MW)   |
| `WInd`            | day-ahead wind generation forecast (MW)    |
| `gas_price`       | global daily gas price, broadcast hourly   |

### Model (`OnionNN`)

```
price_seq, zone_id, hour_id, dow_id, month_id ─┐
                                               ├─ concat → LSTM stack ─┐
   embeddings: zone / hour / dow / month ──────┘                      │
                                                                      ├─ dense head → 24 prices
   onion_probs (B, 24, 4) → ConvOnionFusion (→192-dim) ───────────────┤
   gas_price ─────────────────────────────────────────────────────────┘
```

- Numerical features and the four categorical embeddings are concatenated and
  fed to a multi-layer LSTM (`batch_first = False`, shape `(T, B, F)`).
- The last LSTM timestep is concatenated with the gas price and the
  Onion-fusion output, then passed through a `Linear → GELU → Linear` head that
  emits 24 hourly prices.
- `ConvOnionFusion` takes the per-hour regime probabilities, one-hot-encodes the
  arg-max regime, appends the max confidence, applies a 1-D convolution
  (`5 → 8` channels) + GELU + LayerNorm, and flattens to a 192-dim vector.

The four regime channels are `p_norm` (normal), `p_tail` (tail), `p_neg`
(negative), `p_spike` (spike). They are read from the dataset per zone and
aligned to the 24-hour forecast horizon in `extract_onion_tensor`.

### Training (`OnionTrainer`)

- `OnionTensorDataset` builds sliding windows over each zone's series and
  attaches the matching `(24, 4)` regime tensor at the forecast start.
- Mixed-precision (`autocast` + `GradScaler`) training on CUDA.
- Cosine learning-rate schedule with linear warmup (`cosine_warmup_decay`).
- `EarlyStopping` on validation loss; gradient clipping; NaN-guarded steps.
- `set_trainable(...)` toggles gradients independently for the time embeddings,
  zone embedding, LSTM, and dense head (supports staged / fine-tuning regimes).
- Optuna is available for hyperparameter search.

> **Assumption.** The custom loss passed to the training loop has the signature
> `loss(predictions, target, onion_probs)`, i.e. it is regime-aware. The exact
> loss definition lives in the training notebook rather than in
> `OnionTrainer.py`; consult `Onion-NN.ipynb` for the configuration used to
> produce the shipped weights.

---

## Installation

Requires Python 3.10+

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install torch numpy pandas scikit-learn optuna matplotlib pyarrow scipy
```

A CUDA-capable GPU is recommended; the model and trainer default to
`device="cuda"`. Pass `device="cpu"` to run on CPU (mixed precision is disabled
automatically off-GPU).

> **Note.** The repository does not ship a `requirements.txt`; the list above is
> inferred from the imports in the source files and notebooks.

---

## Usage

### 1. Build the dataset from raw CSVs

```python
import sys; sys.path.append("OnionScripts")
import dataProcessor as dp

dp.clean_csvs("data/raw_data/prices", "data/prices", mode="prices")
dp.clean_csvs("data/raw_data/loads",  "data/loads",  mode="loads")
dp.clean_csvs("data/raw_data/wind_solar", "data/dayahead_wind_solar", mode="wind_solar")

# merge per-zone files, then export one Parquet with gas price attached
dp.processed_csvs_to_parquet(
    input_dir="merged_data",
    gas_filepath="processed_gas.csv",
    output_file="onion_dataset.parquet",
)
```

### 2. Instantiate and train the model

```python
import sys; sys.path.append("OnionScripts")
import torch
from OnionNN import OnionNN
from Embedders import TimestampEmbedder, BZNEmbedder

device = "cuda" if torch.cuda.is_available() else "cpu"

time_emb = TimestampEmbedder(device=device)
bzn_emb  = BZNEmbedder(n_bzn=20, bzn_dim=8, device=device)

model = OnionNN(
    input_size=4,          # numerical features
    hidden_size=128,
    time_embedder=time_emb,
    bzn_embedder=bzn_emb,
    output_size=24,
    device=device,
)
# training_loop(model, optimizer, loss_function, train_loader, val_loader, epochs, device)
```

See `OnionScripts/Onion-NN.ipynb` for the full, runnable training and evaluation
flow (dataset assembly, scaling, Optuna search, prediction collection, plots).

### 3. Load the trained weights

```python
state = torch.load("trained_model/mse/OnionNN_weights.pt", map_location=device)
model.load_state_dict(state)
model.eval()
```

> **Assumption.** `OnionNN_weights.pt` is a `state_dict` saved with the
> hyperparameters used in the training notebook. The model must be constructed
> with matching dimensions before loading.

---

## Results and Outputs

The `trained_model/` directory contains the artifacts of a completed training
run:

- `trained_model/mse/OnionNN_weights.pt` — trained model weights.
- `trained_model/mse/{onion,lstm,lr}_preds.csv` — predictions from the OnionNN
  model and the LSTM / linear-regression baselines, for comparison.
- `trained_model/forecasts/{ZONE}.png` — per-zone forecast-vs-actual plots.
- `trained_model/custom/pred_vs_true_by_bzn.csv` — predicted vs. true prices by
  bidding zone, used by `output_analysis.ipynb`.
- `pics/` — architecture diagrams and error-analysis figures
  (`bzn_errors.png`, `season_errors.png`, …).

> No headline accuracy figures are quoted here on purpose: error metrics depend
> on the train/test split and scaling used in the notebook. Reproduce them with
> `output_analysis.ipynb` against `pred_vs_true_by_bzn.csv`.

---

## Reproducibility

1. Place the required data files in the expected paths. `onion_dataset.parquet`
   and `processed_gas.csv` are included in this repository. The raw per-zone
   CSVs under `data/` originate from ENTSO-E Transparency Platform; verify
   their redistribution terms before publishing or sharing them independently.
2. Open `OnionScripts/Onion-NN.ipynb` and run it end-to-end. The notebook was
   developed in Google Colab (it mounts Google Drive and `pip install`s
   `torch` / `optuna`); when running locally, skip the Drive-mount and
   `sys.path` cells and point the paths at this repository instead.
3. Regenerate prediction dumps and figures, then run `output_analysis.ipynb`
   for the error analysis.

Determinism is not fully pinned (no fixed global seed is set in the scripts), so
exact numbers may vary slightly between runs.

---

## Future Work

- Ship a `requirements.txt` / environment lock and a fixed random seed.
- Move the regime-aware loss and the regime-probability generator out of the
  notebook into versioned source modules.
- Add a non-notebook CLI entry point for training and inference.
- Document and persist the exact hyperparameters bundled with the released
  weights.

---

## Contributors

**Repository owner:** Giorgio Cottini

**Project contributors:** Enrico Paciaroni, Luigi Babiski Arruda, Davide Piccolo

---

## License

This project is licensed under the MIT License.
