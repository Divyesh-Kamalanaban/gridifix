# Gridifix Wiki

**Gridifix** is an end-to-end fault detection and localization engine for medium-voltage (MV) power distribution networks. It pairs a Deep Neural Network (DNN) for normal-state prediction with Random Forest classifiers to detect and locate single-bus faults with high precision.

---

## Navigation

- [Power Grid Basics](Power-Grid-Basics.md) — CIGRE MV Network, pandapower, and distribution grid fundamentals
- [Pipeline Overview](Pipeline-Overview.md) — End-to-end system architecture and workflow explanation
- [Dataset & Features](Dataset-And-Features.md) — Dataset structure, feature engineering, and label definitions
- [Dataflow](Dataflow.md) — Complete data pipeline flow from raw inputs to model outputs
- [Getting Started](Getting-Started.md) — Installation, setup, and quick-start guide

---

## Quick Summary

| Component | Technology | Purpose |
|-----------|-----------|---------|
| DLPF Solver | Custom Python | Distribution Linear Power Flow — replaces Newton-Raphson for 10x+ speedup |
| DNN Baseline | Keras / TensorFlow | Learns normal (healthy) bus voltages & line power flows |
| Residual Engine | NumPy | Computes spatial deviation between live measurements and NN predictions |
| Fault Detection | YDF (Random Forest) | Binary classifier: healthy vs. faulted |
| Fault Localization | YDF (Random Forest) | Multi-class classifier: predicts the faulted line/bus ID |
| Data Synthesizer | pandapower | Generates 21k+ synthetic healthy & faulted grid states |
| MLOps | MLflow | Tracks experiments, metrics, and model artifacts |

---

## Repository

**GitHub:** [Divyesh-Kamalanaban/gridifix](https://github.com/Divyesh-Kamalanaban/gridifix)