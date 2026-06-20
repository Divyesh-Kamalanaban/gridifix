# Pipeline Overview

This page explains Gridifix's end-to-end architecture — from raw power system inputs to a trained, deployable fault detection and localization engine.

---



## 1. System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GRIDIFIX ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT: CIGRE MV Network + Load / DG Variants                        │
│         ↓                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              DATA SYNTHESIZER (pandapower)                   │   │
│  │  • Vary loads (±30%)                                        │   │
│  │  • Place sgens at each bus (bus 1–14, 500 samples each)     │   │
│  │  • Inject faults: randomly take lines out of service         │   │
│  │  • Run pandapower NR solver → ground-truth measurements      │   │
│  │  Output: 21k+ labelled samples                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         ↓                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              DNN TRAINING PIPELINE                           │   │
│  │  • Build dataset: X=[loads, sgens], Y=[bus voltages,        │   │
│  │    line flows]                                               │   │
│  │  • Train Keras DNN to predict Y from X                       │   │
│  │  • Track with MLflow (metrics, checkpoints, artifacts)       │   │
│  │  • Save: model.h5 + input_scaler.pkl + output_scaler.pkl    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         ↓                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              RESIDUAL FEATURE ENGINE                         │   │
│  │  For each sample:                                            │   │
│  │    NN_expected = DNN(X_inputs)                               │   │
│  │    actual      = pandapower measurements (Y_actual)           │   │
│  │    residual_i  = |NN_expected_i − actual_i|                  │   │
│  │  Augment with raw X inputs and raw Y measurements            │   │
│  │  Final feature vector: [X_raw, Y_actual, Y_nn, residuals]   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         ↓                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              CLASSIFIER TRAINING                             │   │
│  │  Model 1 — Fault Detection (Binary YDF/RF):                 │   │
│  │    Input:  Residual feature vector                           │   │
│  │    Output: is_faulted ∈ {0, 1}                               │   │
│  │                                                              │   │
│  │  Model 2 — Fault Localization (Multi-class YDF/RF):         │   │
│  │    Input:  Residual feature vector                           │   │
│  │    Output: fault_location_id ∈ {0, 1, …, 14}                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         ↓                                                           │
│  DEPLOYED MODELS: DNN + Detection RF + Localisation RF            │
│         ↓                                                           │
│  INFERENCE: Live pandapower → residuals → classify → locate       │
└──────────────────────────────────────────────────────────────────────┘
```

---



## 2. Pipeline Stages in Detail

### Stage 1 — Data Synthesis

**File:** [`src/data-synthesis/datasynthesisloop.py`](https://github.com/Divyesh-Kamalanaban/gridifix/blob/main/src/data-synthesis/datasynthesisloop.py)  
**Helper:** [`src/data-synthesis/samplegenerator.py`](https://github.com/Divyesh-Kamalanaban/gridifix/blob/main/src/data-synthesis/samplegenerator.py)

What happens:

1. Load a pre-trained DNN, input scaler, and output scaler.
2. Read the canonical column names from `datasets/combined_power_flow_inputs.csv` and `datasets/combined_power_flow_outputs.csv`.
3. For each of **14 sgen bus locations** (bus 1–14):
   - Generate **500 samples** with random load scaling (±30%) and random sgen placement.
   - For each sample, with **15% probability**, inject a fault by randomly dropping a line.
   - Run `pandapower.runpp()` to get the actual (ground truth) bus voltages and line flows.
   - Pass the same inputs through the DNN to get "expected" (healthy) outputs.
   - Compute residuals (absolute diff between DNN predictions and pandapower measurements).
4. Assemble the feature vector per sample:

   ```
   X = [input_orig_{load/sgen cols},    # raw input values
        actual_output_{vm/p cols},      # pandapower measurements
        nn_pred_normal_{vm/p cols},     # DNN healthy predictions
        residual_{vm/p cols}]           # |actual − predicted|
   ```

5. Assign labels: `is_faulted_label` (0 or 1), `fault_location_id_label` (line ID or −1).

**Output:** `datasets/fault_detection_dataset_full.csv` (~21,000 rows)

### Stage 2 — DNN Training

**File:** [`src/dermodelling.py`](https://github.com/Divyesh-Kamalanaban/gridifix/blob/main/src/dermodelling.py)

The Deep Neural Network is a **multi-layer perceptron (MLP)** that learns the mapping:

```
X_inputs  →  Y_outputs
(loads + sgens)  →  (bus voltages + line power flows)
```

Key design points:

- Takes 28 input features (14 loads × p + q, plus up to 14 sgens × p + q).
- Produces 90 output features (15 buses × [vm_pu, va_degree] + 15 lines × [p_from, q_from, p_to, q_to]).
- Trained with `combined_power_flow_inputs.csv` → `combined_power_flow_outputs.csv` pairs.
- Uses `StandardScaler` for input and output normalization.
- Optimizer: Adam, Loss: Mean Squared Error (MSE).
- Checkpoints and metrics logged to **MLflow** (stored in `mlflow.db` SQLite backend).

**Output:** `models/DNN/model/power_flow_nn_model.h5`, `input_scaler.pkl`, `output_scaler.pkl`

### Stage 3 — Residual Feature Engineering

This is the core innovation that transforms a regression problem into a detection/localization task.

The rationale:

- The DNN learns the mapping between normal operating conditions (loads, sgens) and grid state measurements (voltages, flows).
- After a fault, the mapping is broken; measurements deviate from the learned baseline.
- The **magnitude and spatial pattern** of these deviations characterises both *whether* a fault occurred and *where*.

For inference on a live (or simulated) network:

1. Extract inputs from the pandapower net (loads + sgens).
2. Scale inputs and run the DNN → `y_pred_normal`.
3. Extract actual pandapower results (`res_bus`, `res_line`) → `y_actual`.
4. Scale both with the same output scaler.
5. Compute `residuals = |y_pred_normal − y_actual|`.
6. Build the feature DataFrame for the classifiers.

### Stage 4 — Classifier Training

**Detection Model (Binary):**
- **Algorithm:** Yggdrasil Decision Forest (Gradient Boosted Trees)
- **Target:** `is_faulted_label` (0 = healthy, 1 = faulted)
- **Input:** Residual feature vector (see Stage 3)

**Localization Model (Multi-class):**
- **Algorithm:** Yggdrasil Decision Forest (Gradient Boosted Trees)
- **Target:** `fault_location_id_label` (line ID: 0–14, or −1 for healthy)
- **Input:** Same residual feature vector
- Classes are imbalanced (healthy samples outnumber faulted); handled via sample weighting during training.

### Stage 5 — Online Inference

See [`src/main.py`](https://github.com/Divyesh-Kamalanaban/gridifix/blob/main/src/main.py) for the `predict_fault_status()` function.

Inference flow:

```
1. Receive pandapower network state
2. Build input vector (loads + sgens)
3. Scale → DNN predict → inverse scale → y_pred_normal
4. Extract actual measurements from net.res_bus / net.res_line
5. Scale actual → residuals = |y_pred − y_actual|
6. Build X_for_inference
7. fault_detection_model.predict(X)  →  is_faulted (threshold 0.5)
8. If faulted:
       fault_locate_model.predict(X)  →  probabilities per line
       argmax → predicted_line_id
```

---



## 3. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| DNN for baseline | Captures non-linear grid behaviour; residuals are more discriminative than raw measurements |
| Two-stage RF classifiers | Detection and localization are separate tasks; avoids training a single sparse multi-output model |
| YDF (Gradient Boosted Trees) | Fast inference, robust to feature scaling, handles tabular data well |
| DLPF in pre-processing | Reduces pandapower runtime during data generation (pre-training use only) |
| 15% fault probability | Imbalanced but realistic; detects rare events |
| Constant column removal | Some pandapower outputs (e.g., always-zero line flows for out-of-service lines) are constant and add noise |

---



## 4. MLOps Integration

Gridifix uses **MLflow** for:

- Experiment tracking: hyperparameters, metrics (loss, MAE), and durations.
- Model registry: versioned `.h5` and YDF model artifacts.
- Artifact storage: checkpoints, TensorBoard logs, scaler files.
- Local SQLite backend: `mlflow.db`

To start the MLflow UI:

```bash
mlflow server --port 5000 --backend-store-uri sqlite:///mlflow.db
```

---



## 5. Component Map

| File / Directory | Purpose |
|-----------------|---------|
| `src/data-synthesis/datasynthesisloop.py` | Orchestrates full data generation loop |
| `src/data-synthesis/samplegenerator.py` | Generates individual samples per sgen config |
| `src/dermodelling.py` | DNN architecture, training, and checkpointing |
| `src/dlpf-solver/` | Custom DLPF solver (pre-training speedup) |
| `src/main.py` | Inference entry point; `predict_fault_status()` |
| `models/DNN/` | Saved Keras model and scalers |
| `models/fault_detection_model/` | Trained YDF detection model |
| `models/fault_locate_model/` | Trained YDF localization model |
| `datasets/` | Raw and combined CSVs |
| `mlflow.db` | MLflow experiment metadata |

---



## 6. Summary

Gridifix's pipeline is a **three-phase system**:

1. **Synthesize** — Generate labelled grid states with pandapower.
2. **Learn** — Train a DNN to predict healthy grid behaviour.
3. **Detect & Locate** — Use DNN residuals + YDF classifiers to identify and pinpoint faults.

**Next:** [Dataset & Features](Dataset-And-Features.md) — Understand the exact structure of the data flowing through the pipeline.

**Previous:** [Power Grid Basics](Power-Grid-Basics.md)