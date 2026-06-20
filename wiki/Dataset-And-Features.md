# Dataset & Features

This page details the structure of Gridifix's datasets, the feature engineering pipeline, and the label definitions used to train the detection and localization models.

---



## 1. Datasets Overview

Gridifix uses two primary datasets:

| Dataset | Location | Purpose | Rows | Columns |
|---------|----------|---------|------|---------|
| `combined_power_flow_inputs.csv` | `datasets/` | DNN training inputs (loads + sgens) | ~21,000 | 28 (variable) |
| `combined_power_flow_outputs.csv` | `datasets/` | DNN training targets (voltages + flows) | ~21,000 | 90 (variable) |
| `fault_detection_dataset_full.csv` | `datasets/` | Classifier training features + labels | ~21,000 | 200+ |

> **Note:** Column counts may be lower if constant columns (always-zero features) are removed during preprocessing.

---



## 2. DNN Training Dataset

### 2.1 Input Features (`combined_power_flow_inputs.csv`)

These are the **control variables** — quantities that can be set or measured before running power flow.

```
Format: {load/sgen}_{id}_{quantity}_{unit}

Load columns (always present, 14 loads):
  load_0_p_mw, load_0_q_mvar,
  load_1_p_mw, load_1_q_mvar,
  ...
  load_13_p_mw, load_13_q_mvar

SGen columns (present only when an sgen exists at that bus):
  sgen_bus_{bus_id}_p_mw,
  sgen_bus_{bus_id}_q_mvar
```

**Total:** 28 columns nominally (14 loads × 2 quantities + up to 14 sgens × 2 quantities). In practice, only the sgen buses used during generation appear.

### 2.2 Output Targets (`combined_power_flow_outputs.csv`)

These are the **power flow results** — the quantities the DNN learns to predict.

```
Format: {bus/line}_{id}_{quantity}_{unit}

Bus measurements (15 buses):
  bus_0_vm_pu, bus_0_va_degree,
  bus_1_vm_pu, bus_1_va_degree,
  ...
  bus_14_vm_pu, bus_14_va_degree

Line measurements (15 lines):
  line_0_p_from_mw, line_0_q_from_mvar, line_0_p_to_mw, line_0_q_to_mvar,
  line_1_p_from_mw, line_1_q_from_mvar, line_1_p_to_mw, line_1_q_to_mvar,
  ...
  line_14_p_from_mw, line_14_q_from_mvar, line_14_p_to_mw, line_14_q_to_mvar
```

**Total:** 90 columns (15 buses × 2 + 15 lines × 4).

### 2.3 Constant Column Removal

Some lines may be out of service in certain samples, causing their line measurements to be **always zero** in those subsets. To avoid feeding constant noise into the DNN:

```python
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=constant_cols)
```

This is handled dynamically in both [`samplegenerator.py`](https://github.com/Divyesh-Kamalanaban/gridifix/blob/main/src/data-synthesis/samplegenerator.py) and [`main.py`](https://github.com/Divyesh-Kamalanaban/gridifix/blob/main/src/main.py).

---



## 3. Classifier Training Dataset

### 3.1 Feature Vector Construction

For each sample, the classifier feature vector is assembled as:

```
X_sample = [
    # Block 1: Original inputs (raw)
    input_orig_load_0_p_mw,
    input_orig_load_0_q_mvar,
    input_orig_load_1_p_mw,
    ...,
    input_orig_sgen_bus_{n}_p_mw,
    input_orig_sgen_bus_{n}_q_mvar,

    # Block 2: Actual pandapower outputs (measurements)
    actual_output_bus_0_vm_pu,
    actual_output_bus_0_va_degree,
    actual_output_line_0_p_from_mw,
    actual_output_line_0_q_from_mvar,
    ...,

    # Block 3: DNN predicted outputs (healthy baseline)
    nn_pred_normal_bus_0_vm_pu,
    nn_pred_normal_bus_0_va_degree,
    nn_pred_normal_line_0_p_from_mw,
    nn_pred_normal_line_0_q_from_mvar,
    ...,

    # Block 4: Residuals (absolute difference)
    residual_bus_0_vm_pu,
    residual_bus_0_va_degree,
    residual_line_0_p_from_mw,
    residual_line_0_q_from_mvar,
    ...
]
```

**Rationale for each block:**

| Block | Why include it? |
|-------|-----------------|
| `input_orig_*` | Captures operating condition (load level, DG output) — context for the fault |
| `actual_output_*` | Provides the raw measurement signal before residual computation |
| `nn_pred_normal_*` | Gives the DNN's "expected" healthy state — anchors the residual |
| `residual_*` | The discriminative signal — deviations from normal behaviour |

### 3.2 Labels

| Label Column | Type | Values | Description |
|--------------|------|--------|-------------|
| `is_faulted_label` | Binary (int) | 0 or 1 | 0 = healthy network state; 1 = fault injected |
| `fault_location_id_label` | Integer | −1 or 0–14 | −1 = healthy; 0–14 = line ID where fault occurred |

---



## 4. Data Generation Parameters

Defined in [`src/data-synthesis/datasynthesisloop.py`](https://github.com/Divyesh-Kamalanaban/gridifix/blob/main/src/data-synthesis/datasynthesisloop.py):

```python
num_samples_per_sgen_config = 500      # Samples per sgen bus location
load_variation_range       = (0.7, 1.3)  # ±30% load scaling
sgen_p_mw_range           = (0.0, 0.8)   # SGen active power range
sgen_q_mvar_range_factor  = (-0.5, 0.5) # Q = p_mw * random_factor
fault_probability         = 0.15         # 15% fault injection rate
sgen_buses                = range(1, 15) # 14 bus locations
```

**Expected total:** 14 bus locations × 500 samples = **7,000 samples per sgen config cycle**. Multiple configs iterate to reach the full dataset size.

---



## 5. Feature Scaling

Both DNN inputs and outputs are normalized using **scikit-learn `StandardScaler`**:

```python
from sklearn.preprocessing import StandardScaler

input_scaler  = StandardScaler()  # Fitted on training inputs
output_scaler = StandardScaler()  # Fitted on training outputs

# During inference:
X_scaled = input_scaler.transform(X_raw)
Y_pred   = model.predict(X_scaled)
Y_scaled = output_scaler.transform(Y_actual)
residual  = np.abs(Y_pred - Y_scaled)
```

**Saved artifacts:**
- `models/DNN/model/scalers/input_scaler.pkl`
- `models/DNN/model/scalers/output_scaler.pkl`

---



## 6. Dataset Files

```
datasets/
├── combined_power_flow_inputs.csv   # DNN training inputs
├── combined_power_flow_outputs.csv  # DNN training targets
└── fault_detection_dataset_full.csv # Classifier features + labels
```

**Next:** [Dataflow](Dataflow.md) — Follow the data through each stage of the pipeline.

**Previous:** [Pipeline Overview](Pipeline-Overview.md)