# Getting Started

This page provides step-by-step instructions to set up the Gridifix environment and run the end-to-end pipeline for fault detection and localization.

---



## 1. Prerequisites

Before starting, ensure your system has:

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11 | Recommended (matches `requirements.txt` pins) |
| Conda / Miniconda | Latest | For environment isolation |
| Git | Any | To clone the repository |

---



## 2. Clone the Repository

```bash
git clone https://github.com/Divyesh-Kamalanaban/gridifix.git
cd gridifix
```

---



## 3. Create the Conda Environment

```bash
conda create -n gridifix_py311 python=3.11 pip -y
conda activate gridifix_py311
```

> **Why Python 3.11?**  
> The `requirements.txt` contains pins for TensorFlow, Keras, and other ML libraries that are tested against Python 3.11. Using other versions may cause dependency conflicts.

---



## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | latest | DNN training and inference |
| `keras` | 3.13.2 | High-level neural network API |
| `pandapower` | 3.4.0 | Power flow simulation and data synthesis |
| `pandas` | 2.3.3 | Data manipulation and CSV I/O |
| `numpy` | 2.3.5 | Numerical operations |
| `scikit-learn` | 1.8.0 | StandardScaler and preprocessing |
| `ydf` | latest | Yggdrasil Decision Forests (Random Forest models) |
| `mlflow` | 3.10.1 | Experiment tracking and model registry |
| `joblib` | 1.5.3 | Scaler serialization (.pkl files) |
| `h5py` | 3.14.0 | HDF5 model storage (.h5 files) |

---



## 5. Project Structure

```
gridifix/
├── datasets/
│   ├── combined_power_flow_inputs.csv      # DNN training inputs
│   ├── combined_power_flow_outputs.csv     # DNN training targets
│   └── fault_detection_dataset_full.csv    # Classifier training data
├── docs/assets/                            # Architecture diagrams, charts
├── mlruns/                                 # MLflow experiment logs
├── models/
│   ├── DNN/
│   │   ├── model/
│   │   │   ├── power_flow_nn_model.h5      # Trained Keras DNN
│   │   │   └── scalers/
│   │   │       ├── input_scaler.pkl        # Input normalization
│   │   │       └── output_scaler.pkl       # Output normalization
│   │   └── scripts/
│   │       └── powerflowdnn.py             # DNN training script
│   ├── fault_detection_model/              # YDF binary classifier
│   └── fault_locate_model/                 # YDF multi-class classifier
├── src/
│   ├── data-synthesis/
│   │   ├── datasynthesisloop.py            # Data synthesis orchestrator
│   │   ├── initcigre.py                    # CIGRE network initialization
│   │   ├── samplegenerator.py              # Individual sample generation
│   │   └── [dlpf-solver scripts]           # DLPF coefficient/solver
│   ├── dlpf-solver/
│   │   ├── dlpfcoeff.py
│   │   ├── solvedlpf.py
│   │   └── erroranalysisdlpf.py
│   ├── dermodelling.py                     # DNN architecture & training
│   ├── main.py                             # Inference entry point
│   └── results/
│       └── time-comparison-nn.py
├── .gitignore
├── LICENSE                                 # Apache 2.0
├── README.md
├── requirements.txt
└── mlflow.db                               # MLflow SQLite backend
```

---



## 6. Quick Start: Run Inference

If the pre-trained models are already present in `models/`, you can run inference immediately:

```bash
python src/main.py
```

**Expected output:**

```
Successfully loaded power flow NN model, its scalers and the Random Forest Regressors in main script.

Simulating a faulted network state
prediction: 4
probability: 0.87
Inference Result (Faulted on Line 4): {'is_faulted': 1, 'fault_localization': {...}}
```

---



## 7. Full Pipeline: Retrain from Scratch

### 7.1 Train the DNN

If you want to retrain the DNN on new data (or the existing CSVs):

```bash
python src/dermodelling.py
```

This will:
1. Load `combined_power_flow_inputs.csv` and `combined_power_flow_outputs.csv`.
2. Build and compile a Keras MLP.
3. Train with Adam optimizer and MSE loss.
4. Log metrics to MLflow.
5. Save:
   - `models/DNN/model/power_flow_nn_model.h5`
   - `models/DNN/model/scalers/input_scaler.pkl`
   - `models/DNN/model/scalers/output_scaler.pkl`

### 7.2 Generate Classifier Training Data

```bash
python src/data-synthesis/datasynthesisloop.py
```

This will:
1. Load the trained DNN and scalers.
2. For each of 14 sgen bus locations, generate 500 samples.
3. Inject faults at 15% probability.
4. Compute residuals and assemble feature vectors.
5. Save to `datasets/fault_detection_dataset_full.csv`.

**Logs to expect:**

```
Total Samples Generated: 7000
Final DataFrame shape: (7000, ~200)
Value counts for 'is_faulted_label':
0    5950
1    1050
```

### 7.3 Train the YDF Classifiers

The YDF models are trained separately (typically in the same script or a dedicated training notebook). The training code uses `fault_detection_dataset_full.csv` to fit:

- `fault_detection_model` (binary classification, `is_faulted_label`).
- `fault_locate_model` (multi-class classification, `fault_location_id_label`).

Once trained, save them to:
- `models/fault_detection_model/content/fault_detection_model/`
- `models/fault_locate_model/content/fault_locate_model/`

---



## 8. Track Experiments with MLflow

Start the MLflow UI to view training metrics, model versions, and artifacts:

```bash
mlflow server \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Then open `http://localhost:5000` in your browser.

### What Gets Logged

| Item | Details |
|------|---------|
| Parameters | Learning rate, batch size, epoch count, layer sizes |
| Metrics | Training/validation loss, MAE per epoch |
| Artifacts | Model `.h5`, scalers `.pkl`, TensorBoard logs |
| Runs | Unique run IDs (e.g., `bemused-fly-207`) |

---



## 9. Verify Installation

Run the following to confirm all imports work:

```bash
python -c "
import pandapower as pp
from pandapower.networks import create_cigre_network_mv
import tensorflow as tf
import ydf
import pandas as pd
import numpy as np
import joblib
print('All core dependencies imported successfully.')
net = create_cigre_network_mv()
pp.runpp(net)
print(f'CIGRE MV network loaded. Converged: {net.converged}')
print(f'Buses: {len(net.bus)}, Lines: {len(net.line)}, Loads: {len(net.load)}')
"
```

**Expected output:**

```
All core dependencies imported successfully.
CIGRE MV network loaded. Converged: True
Buses: 15, Lines: 15, Loads: 14
```

---



## 10. Common Issues & Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` for `.h5` or `.pkl` files | Models not yet trained | Run `python src/dermodelling.py` first |
| `Power Flow Not Converged` | Unrealistic load/sgen values | Clamp inputs to valid ranges |
| `ImportError: No module named 'ydf'` | YDF not installed | `pip install ydf` |
| MLflow UI shows no runs | Backend URI misconfigured | Use `--backend-store-uri sqlite:///mlflow.db` |
| Slow pandapower runtime | Large network or high iteration count | Use DLPF solver for preprocessing |
| Constant column mismatch | CSVs regenerated with different samples | Re-run data synthesis to regenerate CSVs |

---



## 11. Next Steps

- Explore [Power Grid Basics](Power-Grid-Basics.md) to understand the CIGRE MV network.
- Read [Pipeline Overview](Pipeline-Overview.md) for the system architecture.
- Review [Dataset & Features](Dataset-And-Features.md) and [Dataflow](Dataflow.md) for implementation details.

---

**Previous:** [Dataflow](Dataflow.md)