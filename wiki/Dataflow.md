# Dataflow

This page traces a single data point — from raw CIGRE network creation through to a final fault prediction — showing exactly how data moves between files, functions, and model artifacts.

---



## 1. High-Level Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA SYNTHESIS                                             │
│                                                                    │
│  create_cigre_network_mv()                                         │
│         ↓                                                          │
│  [Vary loads ±30%, add sgen, maybe fault a line]                   │
│         ↓                                                          │
│  pp.runpp() → actual voltages & flows                              │
│         ↓                                                          │
│  DNN.predict() → predicted voltages & flows                        │
│         ↓                                                          │
│  residuals = |actual − predicted|                                  │
│         ↓                                                          │
│  X_feature_vector = [inputs_raw, actual, predicted, residuals]     │
│  y_label = [is_faulted, fault_location_id]                         │
│         ↓                                                          │
│  Save to fault_detection_dataset_full.csv                          │
└────────────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 2: TRAINING                                                  │
│                                                                    │
│  combined_power_flow_inputs.csv  ──┐                               │
│  combined_power_flow_outputs.csv  ──┤──→ DNN.fit()                │
│                                    │    (Inputs → Outputs)         │
│  fault_detection_dataset_full.csv ──┤                               │
│                                    │                               │
│                                    ├──→ fault_detection_model.fit()│
│                                    │    (Features → is_faulted)    │
│                                    │                               │
│                                    └──→ fault_locate_model.fit()   │
│                                         (Features → line_id)      │
└────────────────────────────────────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 3: INFERENCE                                                 │
│                                                                    │
│  pandapower network (live / simulated)                             │
│         ↓                                                          │
│  Extract loads & sgens → X_inputs_raw                              │
│         ↓                                                          │
│  input_scaler.transform(X_inputs_raw)                              │
│         ↓                                                          │
│  DNN.predict() → Y_pred_scaled                                    │
│         ↓                                                          │
│  output_scaler.inverse_transform(Y_pred_scaled)                    │
│         ↓                                                          │
│  Y_pred_original = nn_predicted_outputs_original                   │
│         ↓                                                          │
│  Extract net.res_bus / net.res_line → Y_actual_original           │
│         ↓                                                          │
│  output_scaler.transform(Y_actual_original)                        │
│         ↓                                                          │
│  Y_actual_scaled                                                   │
│         ↓                                                          │
│  residuals_scaled = |Y_pred_scaled − Y_actual_scaled|             │
│         ↓                                                          │
│  X_for_inference = [                                               │
│      input_orig_*, actual_output_*,                                │
│      nn_pred_normal_*, residual_*                                  │
│  ]                                                                 │
│         ↓                                                          │
│  fault_detection_model.predict(X) → is_faulted                     │
│         ↓ (if faulted)                                             │
│  fault_locate_model.predict(X) → probabilities per line            │
│         ↓                                                          │
│  argmax(probabilities) → predicted_line_id                         │
└────────────────────────────────────────────────────────────────────┘
```

---



## 2. Detailed Data Trace: One Sample

Follow one sample through the entire pipeline at the code level.

### Step 1 — Network Construction

**File:** `src/data-synthesis/samplegenerator.py`  
**Function:** `generate_samples()`

```python
# Line 30-34
base_net = create_cigre_network_mv()
pp.runpp(base_net)
orig_loads = base_net.load.copy()
all_lines = base_net.line.index.to_list()
```

- `create_cigre_network_mv()` returns a `pandapowerNet` with:
  - 15 buses (bus 0 = slack)
  - 15 lines
  - 14 loads
  - Default sgen configuration (may be empty initially)

### Step 2 — Input Perturbation

**File:** `src/data-synthesis/samplegenerator.py`  
**Lines 58-86**

```python
# Vary loads by ±30%
for idx, load_rows in orig_loads.iterrows():
    p_scale = np.random.uniform(0.7, 1.3)
    q_scale = np.random.uniform(0.7, 1.3)
    temp_net.load.at[idx, 'p_mw'] = load_rows['p_mw'] * p_scale
    temp_net.load.at[idx, 'q_mvar'] = load_rows['q_mvar'] * q_scale
    current_inputs_dict[f'load_{idx}_p_mw'] = temp_net.load.at[idx, 'p_mw']
    current_inputs_dict[f'load_{idx}_q_mvar'] = temp_net.load.at[idx, 'q_mvar']
```

- `current_inputs_dict` accumulates raw input values keyed by feature name.

### Step 3 — SGen Injection

**File:** `src/data-synthesis/samplegenerator.py`  
**Lines 69-83**

```python
# Remove existing sgen at target bus, add new random sgen
sgen_bus = sgen_config['bus']
pp.create_sgen(temp_net, bus=sgen_bus, p_mw=p_sgen, q_mvar=q_sgen)
current_inputs_dict[f'sgen_bus_{sgen_bus}_p_mw'] = p_sgen
current_inputs_dict[f'sgen_bus_{sgen_bus}_q_mvar'] = q_sgen
```

- SGen is placed at one of the 14 bus locations (bus 1–14).
- `sgen_config` dict comes from the iteration loop in `datasynthesisloop.py`.

### Step 4 — Fault Injection (Probabilistic)

**File:** `src/data-synthesis/samplegenerator.py`  
**Lines 94-105**

```python
is_faulted = 0
fault_location_id = -1

if np.random.rand() < fault_probability:  # 15% chance
    is_faulted = 1
    faulted_line_id = np.random.choice(all_lines)
    temp_net.line.at[faulted_line_id, 'in_service'] = False
    fault_location_id = faulted_line_id
```

- With 15% probability, a random line is taken out of service (simulating a single-bus fault).
- If no fault: `is_faulted = 0`, `fault_location_id = -1`.
- If fault: `is_faulted = 1`, `fault_location_id = line_id (0-14)`.

### Step 5 — Ground-Truth Power Flow

**File:** `src/data-synthesis/samplegenerator.py`  
**Lines 107-128**

```python
pp.runpp(temp_net)
full_actual_outputs_list = []
for id in bus_ids:
    full_actual_outputs_list.extend([
        temp_net.res_bus.at[id, 'vm_pu'],
        temp_net.res_bus.at[id, 'va_degree']
    ])
for id in line_ids:
    full_actual_outputs_list.extend([
        temp_net.res_line.at[id, 'p_from_mw'],
        temp_net.res_line.at[id, 'q_from_mvar'],
        temp_net.res_line.at[id, 'p_to_mw'],
        temp_net.res_line.at[id, 'q_to_mvar']
    ])
```

- `pp.runpp()` uses Newton-Raphson internally.
- Results are stored in `temp_net.res_bus` and `temp_net.res_line`.
- 90-element list: 15 buses × 2 + 15 lines × 4.

### Step 6 — DNN Prediction (Healthy Baseline)

**File:** `src/data-synthesis/samplegenerator.py`  
**Lines 86-91**

```python
nn_input_df_row = pd.DataFrame([current_inputs_dict], columns=nn_input_columns).fillna(0)
scaled_nn_inputs = input_scaler.transform(nn_input_df_row)
nn_predicted_outputs_scaled = nn_model.predict(scaled_nn_inputs, verbose=0)
nn_predicted_outputs_original = output_scaler.inverse_transform(nn_predicted_outputs_scaled)
```

- Inputs are scaled with `input_scaler` (fitted during DNN training).
- DNN outputs are in scaled space; inverse-transformed to original units.
- `nn_predicted_outputs_original[0]` is a 90-element array: the DNN's guess of what the grid *should* look like.

### Step 7 — Residual Computation

**File:** `src/data-synthesis/samplegenerator.py`  
**Lines 137-139**

```python
scaled_actual_outputs = output_scaler.transform(actual_outputs_for_scaler)
residuals = np.abs(nn_predicted_outputs_scaled - scaled_actual_outputs)
```

- Both actual and predicted outputs are scaled to the same space.
- `residuals[0]` is a 90-element array of absolute differences.
- Large residuals indicate the DNN's "healthy" model doesn't match reality → likely a fault.

### Step 8 — Feature Vector Assembly

**File:** `src/data-synthesis/samplegenerator.py`  
**Lines 142-166**

```python
record = {}
# Block 1: Original inputs
for col_name in nn_input_columns:
    record[f'input_orig_{col_name}'] = current_inputs_dict.get(col_name, 0.0)

# Block 2: Actual outputs (ground truth)
for col_idx, col_name in enumerate(nn_output_columns):
    record[f'actual_output_{col_name}'] = actual_outputs_for_scaler[0, col_idx]

# Block 3: DNN predicted outputs
for col_idx, col_name in enumerate(nn_output_columns):
    record[f'nn_pred_normal_{col_name}'] = nn_predicted_outputs_original[0, col_idx]

# Block 4: Residuals
for col_idx, col_name in enumerate(nn_output_columns):
    record[f'residual_{col_name}'] = residuals[0, col_idx]

# Labels
record['is_faulted_label'] = is_faulted
record['fault_location_id_label'] = fault_location_id
```

- `record` is a single dict representing one sample.
- Appended to `combined_dataset_records` list.
- Final DataFrame: `pd.DataFrame(combined_dataset_records)`.

### Step 9 — Dataset Persistence

**File:** `src/data-synthesis/datasynthesisloop.py`  
**Lines 145-166**

```python
final_detection_dataset_df = pd.concat(all_generated_data_frames, ignore_index=True)
final_detection_dataset_df.to_csv("fault_detection_dataset_full.csv", index=False)
```

- 14 sgen configs × 500 samples = **7,000** samples per iteration cycle.
- Saved to `datasets/fault_detection_dataset_full.csv`.

---



## 3. Inference Data Flow

When deploying the trained models for real-time or simulated fault detection.

### 3.1 Entry Point

**File:** `src/main.py`  
**Function:** `predict_fault_status(current_net: pp.pandapowerNet)`

```python
def predict_fault_status(current_net: pp.pandapowerNet) -> dict:
```

- Takes a **fully-solved pandapower network** (`current_net.runpp()` must have been called and converged).

### 3.2 Extract Live Inputs

**File:** `src/main.py`  
**Lines 64-88**

```python
current_inputs_dict = {}

for idx, load_rows in current_net.load.iterrows():
    current_inputs_dict[f'load_{idx}_p_mw'] = load_rows['p_mw']
    current_inputs_dict[f'load_{idx}_q_mvar'] = load_rows['q_mvar']

# Default sgen values to 0.0 for missing columns
for col in original_nn_input_columns:
    if 'sgen_bus_' in col and col not in current_inputs_dict:
        current_inputs_dict[col] = 0.0

# Aggregate sgen power at each bus
sgen_power_at_bus = current_net.sgen.groupby('bus')[['p_mw', 'q_mvar']].sum().to_dict('index')
for bus_id, values in sgen_power_at_bus.items():
    current_inputs_dict[f'sgen_bus_{bus_id}_p_mw'] = values['p_mw']
    current_inputs_dict[f'sgen_bus_{bus_id}_q_mvar'] = values['q_mvar']

nn_input_df_row = pd.DataFrame([current_inputs_dict], columns=original_nn_input_columns).fillna(0)
```

- Mirrors the exact extraction logic from `samplegenerator.py` for consistency.

### 3.3 DNN Inference

**File:** `src/main.py`  
**Lines 92-98**

```python
scaled_nn_inputs = loaded_input_scaler.transform(nn_input_df_row)
nn_predicted_outputs_scaled = loaded_nn_model.predict(scaled_nn_inputs, verbose=0)

nn_predicted_outputs_scaled_df = pd.DataFrame(nn_predicted_outputs_scaled,
                                              columns=original_nn_output_columns)
nn_predicted_outputs_original = loaded_output_scaler.inverse_transform(
    nn_predicted_outputs_scaled_df)
nn_predicted_outputs_original_df = pd.DataFrame(nn_predicted_outputs_original,
                                                columns=original_nn_output_columns)
```

- `loaded_nn_model` → `models/DNN/model/power_flow_nn_model.h5`
- `loaded_input_scaler` → `models/DNN/model/scalers/input_scaler.pkl`
- `loaded_output_scaler` → `models/DNN/model/scalers/output_scaler.pkl`

### 3.4 Extract Actual Measurements

**File:** `src/main.py`  
**Lines 103-133**

```python
actual_outputs_from_pandapower_list = []
bus_results = current_net.res_bus
line_results = current_net.res_line

for col_name in original_nn_output_columns:
    value = 0.0
    if col_name.startswith('bus_'):
        bus_id = int(col_name.split('_')[1])
        measure_type = '_'.join(col_name.split('_')[2:])
        value = bus_results.at[bus_id, measure_type]
    elif col_name.startswith('line_'):
        line_id = int(col_name.split('_')[1])
        measure_type = '_'.join(col_name.split('_')[2:])
        if line_id in line_results.index:
            value = line_results.at[line_id, measure_type]
    actual_outputs_from_pandapower_list.append(value)

actual_outputs_from_pandapower_df = pd.DataFrame(
    [actual_outputs_from_pandapower_list], columns=original_nn_output_columns)
```

- Gracefully handles out-of-service lines (returns 0.0 if line not in results).

### 3.5 Residual Computation

**File:** `src/main.py`  
**Lines 141-148**

```python
scaled_actual_outputs = loaded_output_scaler.transform(actual_outputs_from_pandapower_df)
residuals_np = np.abs(nn_predicted_outputs_scaled - scaled_actual_outputs)

residual_feature_names = [f'residual_{col}' for col in original_nn_output_columns]
residuals_df = pd.DataFrame(residuals_np, columns=residual_feature_names)
```

### 3.6 Build Inference Feature DataFrame

**File:** `src/main.py`  
**Lines 150-170**

```python
X_for_inference = pd.DataFrame(index=[0])

# Add original inputs
for col in original_nn_input_columns:
    X_for_inference[f'input_orig_{col}'] = current_inputs_dict.get(col, 0.0)

# Add actual outputs
for col in original_nn_output_columns:
    X_for_inference[f'actual_output_{col}'] = actual_outputs_from_pandapower_df[col].iloc[0]

# Add predicted outputs
for col in original_nn_output_columns:
    X_for_inference[f'nn_pred_normal_{col}'] = nn_predicted_outputs_original_df[col].iloc[0]

# Add residuals
for col in residual_feature_names:
    X_for_inference[col] = residuals_df[col].iloc[0]

# Remove label columns if any snuck in
feature_columns = [col for col in X_for_inference.columns if not col.endswith('_label')]
X_for_inference = X_for_inference[feature_columns]
```

### 3.7 Fault Detection

**File:** `src/main.py`  
**Lines 172-183**

```python
detection_res = fault_detection_model.predict(X_for_inference)
# detection_res is a probability (0-1)
is_faulted = 1 if detection_res >= 0.5 else 0
result = {'is_faulted': is_faulted}
```

- `fault_detection_model` → `models/fault_detection_model/content/fault_detection_model` (YDF)
- Threshold: 0.5 (configurable).

### 3.8 Fault Localization

**File:** `src/main.py`  
**Lines 185-210**

```python
if is_faulted == 1:
    locate_res = fault_locate_model.predict(X_for_inference)
    # locate_res[0] is an array of probabilities per line class
    fault_locate_label_ids = fault_locate_model.label_classes()
    class_labels = [int(x) for x in fault_locate_label_ids]

    predicted_localization_probs = {}
    for i, prob in enumerate(locate_res[0]):
        predicted_localization_probs[class_labels[i]] = prob

    most_likely_fault_id = max(predicted_localization_probs,
                               key=predicted_localization_probs.get)
    most_likely_fault_prob = predicted_localization_probs[most_likely_fault_id]

    result["fault_localization"] = {
        "predicted_line_id": most_likely_fault_id,
        "probability": most_likely_fault_prob,
        "all_probabilities": predicted_localization_probs
    }
```

- `fault_locate_model` → `models/fault_locate_model/content/fault_locate_model` (YDF)
- Returns both the predicted line ID and the full probability distribution.

---



## 4. Data Shape Summary

| Stage | Object | Shape / Type |
|-------|--------|--------------|
| Raw pandapower network | `pp.pandapowerNet` | — |
| Raw inputs | `current_inputs_dict` | dict: 28 keys |
| Input DataFrame | `nn_input_df_row` | (1, N_input_cols) |
| Scaled inputs | `scaled_nn_inputs` | (1, N_input_cols) |
| DNN scaled outputs | `nn_predicted_outputs_scaled` | (1, 90) |
| DNN original outputs | `nn_predicted_outputs_original` | (1, 90) |
| Actual measurements | `actual_outputs_from_pandapower_list` | list: 90 floats |
| Scaled actual outputs | `scaled_actual_outputs` | (1, 90) |
| Residuals | `residuals_np` | (1, 90) |
| Inference features | `X_for_inference` | (1, ~200) |
| Detection probability | `detection_res` | float: 0-1 |
| Localization probabilities | `locate_res[0]` | (15,) — one per line class |

---



## 5. Key Files in Data Flow

| File | Role in Data Flow |
|------|-------------------|
| `src/data-synthesis/datasynthesisloop.py` | Orchestrates multi-config data synthesis; persists final CSV |
| `src/data-synthesis/samplegenerator.py` | Generates one sample: perturbs, faults, runs PF, computes residuals |
| `src/dermodelling.py` | Trains DNN on inputs/outputs CSVs; saves model + scalers |
| `src/main.py` | `predict_fault_status()`: complete inference pipeline |
| `datasets/combined_power_flow_inputs.csv` | DNN training input features |
| `datasets/combined_power_flow_outputs.csv` | DNN training target features |
| `datasets/fault_detection_dataset_full.csv` | Classifier training data |
| `models/DNN/model/power_flow_nn_model.h5` | Trained Keras DNN |
| `models/DNN/model/scalers/input_scaler.pkl` | Input normalization |
| `models/DNN/model/scalers/output_scaler.pkl` | Output normalization |
| `models/fault_detection_model/` | YDF binary classifier |
| `models/fault_locate_model/` | YDF multi-class classifier |

---



## 6. Summary

Every sample follows the same path:

1. **Construct / Load** → pandapower network
2. **Perturb** → randomize loads and sgens
3. **Fault?** → probabilistically drop a line
4. **Solve** → `pp.runpp()` for ground truth
5. **Predict** → DNN computes expected healthy state
6. **Residual** → absolute difference between expected and actual
7. **Feature Vector** → concatenate inputs, actuals, predictions, residuals
8. **Label** → `is_faulted`, `fault_location_id`
9. **Classify** → YDF models detect and locate

**Next:** [Getting Started](Getting-Started.md) — Set up and run the pipeline yourself.

**Previous:** [Dataset & Features](Dataset-And-Features.md)