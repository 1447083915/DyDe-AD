# DyDe-AD: Dynamic Decomposition and Dual-Domain Anomaly Detection for UAV Telemetry

> **From Signal Decomposition to Dynamical Decoupling: A Mori–Zwanzig Perspective for UAV Anomaly Detection**

Official implementation of the DyDe-AD framework — a decomposition-before-detection approach for in-flight UAV anomaly detection.

---

## Overview

Reliable in-flight anomaly detection is critical for UAVs, but raw telemetry couples **control-induced flight dynamics** with **disturbance-related residual patterns**. Existing methods either model temporal deviations (missing global disturbances) or directly apply time–frequency decomposition (introducing pseudo-structural spectral noise from control responses).

**DyDe-AD** addresses both challenges through a three-stage pipeline:

```
Raw Telemetry  →  DeepKoopman Decoupling  →  (M, P)
                      │                         │
                      │           ┌─────────────┴──────────────┐
                      │           │                            │
                      │     MRA-Rec (temporal)         Morlet-VAE (spectral)
                      │           │                            │
                      │           └─────────────┬──────────────┘
                      │                         │
                      ▼                         ▼
               Markov-dominant M          Memory-kernel P
               reconstruction error       spectral reconstruction error
                      │                         │
                      └─────────┬───────────────┘
                                ▼
                         Anomaly Score
```

### Key Components

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | **DeepKoopman** | Mori–Zwanzig-guided Koopman autoencoder that decouples telemetry into a Markov-dominant component **M** and a memory-kernel component **P** |
| 2 | **MRA-Rec** | Multi-scale Retrieval-Augmented Temporal Reconstructor — reconstructs **M** using scale-specific prototype retrieval with dilated convolutions at rates {1,2,4,8} |
| 3 | **Morlet-VAE** | Morlet-wavelet variational autoencoder — reconstructs **P** in the spectral domain for frequency-aware anomaly detection |

---

## MRA-Rec Architecture

MRA-Rec (Multi-scale Retrieval-Augmented Temporal Reconstructor) is the temporal reconstruction branch. It replaces standard autoencoders with a prototype-based retrieval mechanism that preserves multi-scale flight dynamics:

### 1. Multi-scale Sequential Encoding
Four parallel 1-D convolutional branches with dilation rates `R = {1, 2, 4, 8}` extract temporal features at different receptive fields:

$$H_r = \text{GELU}(\text{Conv1d}_r(M)), \quad r \in \{1, 2, 4, 8\}$$

- Small-dilation branches capture rapid control responses (attitude corrections, motor speed changes)
- Large-dilation branches capture manoeuvre-level regimes (hovering, cruising, loitering)

### 2. Multi-scale Prototype Learning
Each scale maintains an independent prototype codebook $P_r = \{p_{r,1}, \dots, p_{r,N_r}\} \in \mathbb{R}^{N_r \times c}$ (32 prototypes/scale). Continuous features are quantised to their nearest prototype:

$$e_{r,t} = p_{r,k^*}, \quad k^* = \arg\min_k \|h_{r,t} - p_{r,k}\|^2$$

The VQ commitment loss pulls features and prototypes toward each other:

$$\mathcal{L}^r_{vq} = \sum_t \|\text{sg}[h_{r,t}] - e_{r,t}\|^2 + \beta \|h_{r,t} - \text{sg}[e_{r,t}]\|^2$$

### 3. Scale-indexed Cross Retrieval
Learnable **scale embeddings** $s_r \in \mathbb{R}^c$ are injected into queries and keys so the attention mechanism distinguishes temporal granularities:

- **Query:** $Q_r = (\tilde{H}_r + \mathbf{1}_T \otimes s_r) W_{\text{query}}$
- **Key:** $K_r = (P_r + s_r) W_{\text{key}}$ (all prototypes across all scales)
- **Value:** $V_r = P_r W_{\text{value}}$
- **Retrieval:** $R_r = \text{softmax}(Q_r K^\top / \sqrt{c}) V$
- **Output:** $\hat{H}_r = \tilde{H}_r + R_r$ (residual connection)

### 4. Reconstruction
The enhanced multi-scale representations are concatenated and decoded:

$$\hat{M} = D_T\big([\hat{H}_1 \,\|\, \hat{H}_2 \,\|\, \hat{H}_4 \,\|\, \hat{H}_8]\big)$$

where $D_T$ is a lightweight 2-layer MLP.

---

## Project Structure

```
DyDe-AD/
├── run.py                          # Main entry point
├── requirements.txt                # Python dependencies
├── models/
│   ├── __init__.py
│   └── Ours.py                     # DyDe-AD model (DeepKoopman + MRA-Rec + WaveletVAE)
├── exp/
│   ├── __init__.py
│   ├── exp_basic.py                # Base experiment class
│   └── exp_anomaly_detection.py    # Training + evaluation (5-fold CV)
├── data_provider/
│   ├── __init__.py
│   ├── data_factory.py             # Data provider factory
│   └── data_loader.py              # UAVSegLoader (windowed segment dataloader)
├── utils/
│   ├── __init__.py
│   ├── tools.py                    # EarlyStopping, learning rate scheduler, metric adjustment
│   └── print_args.py               # Argument logging
├── dataset/                        # Data preprocessing scripts
│   ├── RflyMAD/                    # RflyMAD dataset helpers
│   └── ALFA/                       # ALFA dataset helpers
├── Paper/
│   └── DyDe-AD_WISE.pdf            # Full paper
└── checkpoints/                    # Saved model weights (created at runtime)
```

---

## Quick Start

### Installation

```bash
# Python ≥ 3.8
pip install -r requirements.txt
```

### Data Preparation

Place your UAV telemetry data in the following structure:

```
dataset/
└── {DATASET_NAME}/
    ├── train/
    │   ├── flight_01.csv
    │   ├── flight_02.csv
    │   └── ...
    ├── test/
    │   ├── flight_01.csv
    │   └── ...
    └── test_label/
        ├── flight_01_label.csv
        └── ...
```

Each CSV contains telemetry readings with the first column as timestamps and subsequent columns as features. The first `control_dim` columns are treated as control inputs and the remaining as state observations.

### Training & Evaluation

```bash
python run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --model Ours \
    --model_id DyDeAD_RflyMAD \
    --data RflyMAD \
    --root_path ./dataset/RflyMAD/ \
    --seq_len 175 \
    --enc_in 23 \
    --d_model 128 \
    --latent_dim 64 \
    --batch_size 128 \
    --train_epochs 100 \
    --learning_rate 0.001 \
    --patience 3 \
    --lradj cosine \
    --anomaly_ratio 0.25
```

Key arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Dataset name (`RflyMAD`, `ALFA`, `UAV_RFD`) | `RflyMAD` |
| `--seq_len` | Sliding window length | `175` |
| `--enc_in` | Total input features (control + state) | dataset-dependent |
| `--d_model` | Hidden dimension | `128` |
| `--latent_dim` | Koopman latent dimension (~1.5× state_dim) | `64` |
| `--train_epochs` | Total training epochs (Stage 1 + Stage 2) | `100` |
| `--batch_size` | Batch size | `128` |

### Two-Stage Training

The training is automatically split into two stages:

- **Stage 1** (first half of epochs): Train only the DeepKoopman decomposition module. Objective: reconstruction + one-step Markov latent dynamics.
- **Stage 2** (second half): Freeze the decomposition module; jointly train MRA-Rec and Morlet-VAE on the fixed **M** and **P** components.

---

## Model Configuration

### DeepKoopman Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `latent_dim` | Koopman latent space dimension | 64 (RflyMAD) / 48 (ALFA) |
| `d_model` | Hidden layer width | 128 |
| Control columns | First N features treated as control input | 8 (RflyMAD) / 5 (ALFA) / 0 (other) |

### MRA-Rec Parameters

| Parameter | Description | Value |
|-----------|-------------|-------|
| Dilation rates | Multi-scale receptive fields | {1, 2, 4, 8} |
| Prototypes per scale | Codebook size per scale $N_r$ | 32 |
| Commitment cost $\beta$ | VQ commitment loss weight | 0.25 |
| `mem_dim` | Prototype embedding dimension | 128 |

### Morlet-VAE Parameters

| Parameter | Value |
|-----------|-------|
| Morlet scales | {1, 2, 4, 8} |
| Central frequency $\omega_0$ | 5.0 |
| VAE latent dimension | 64 |

---

## Performance

Results on real-world and simulated UAV datasets compared against 17 representative baselines:

| Metric | Description |
|--------|-------------|
| **F1 Score** | Best-F1 threshold search on [0%, 25%] anomaly ratio |
| **AUC** | Area under ROC curve (no threshold needed) |
| **TTD** | Time-to-Detection (steps within anomaly segment) |
| **MR** | Miss Rate (fraction of undetected anomaly segments) |

DyDe-AD achieves competitive or superior performance with only **~0.47M trainable parameters** and the **highest inference throughput** among all compared methods.

---

## Citation

```bibtex
@inproceedings{dydead2025,
  title     = {From Signal Decomposition to Dynamical Decoupling:
               A Mori–Zwanzig Perspective for UAV Anomaly Detection},
  author    = {Anonymous},
  booktitle = {Under Review},
  year      = {2025}
}
```

## License

See [LICENSE](LICENSE) for details.

## Acknowledgements

This codebase builds upon the [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) framework.
