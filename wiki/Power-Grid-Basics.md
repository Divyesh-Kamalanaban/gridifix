# Power Grid Basics

This page covers the fundamental power system concepts, the CIGRE MV benchmark network, and the `pandapower` simulation framework that underpin Gridifix.

---



## 1. What Is a Distribution Network?

A **Medium-Voltage (MV) distribution network** typically operates between **1 kV and 35 kV**. Its purpose is to deliver electrical power from transmission/sub-transmission networks to end consumers (residential, commercial, and industrial loads). Key characteristics include:

- **Radial or Meshed Topology:** Most MV networks are radial (tree-like), though some urban grids use meshed layouts for reliability.
- **High R/X Ratio:** Distribution lines have relatively high resistance compared to reactance, making voltage drop a dominant concern.
- **Unbalanced Operation:** Loads are often single-phase, causing voltage and current unbalance across the three phases.
- **Distributed Generation (DG):** PV inverters, small hydro, and wind turbines (modelled as `sgen` in pandapower) are increasingly connected at MV buses.

---



## 2. Key Power Flow Concepts

### 2.1 Power Flow (Load Flow) Analysis

Power flow analysis solves for the **steady-state voltages** at every bus and the **real/reactive power flows** on every line, given:

- Known bus injections (loads, generators, shunts)
- Network topology and line/cable parameters

Gridifix uses power flow results as both **training targets** (for the DNN) and **real-time measurements** (for residual computation).

### 2.2 Bus Types

| Bus Type | Symbol | Known | Unknown |
|----------|--------|-------|---------|
| Slack / Reference | `sl_bus` | V magnitude, V angle | P, Q injections |
| PV (Generator) | `pv_bus` | P injection, V magnitude | Q injection, V angle |
| PQ (Load) | `pq_bus` | P injection, Q injection | V magnitude, V angle |

### 2.3 Key Measurements

| Quantity | Symbol | Unit | Description |
|----------|--------|------|-------------|
| Voltage Magnitude | `vm_pu` | per unit (pu) | Bus voltage relative to nominal |
| Voltage Angle | `va_degree` | degrees | Phase angle of bus voltage |
| Active Power Flow (from) | `p_from_mw` | MW | Active power leaving "from" end of line |
| Reactive Power Flow (from) | `q_from_mvar` | MVAr | Reactive power leaving "from" end |
| Active Power Flow (to) | `p_to_mw` | MW | Active power arriving at "to" end |
| Reactive Power Flow (to) | `q_to_mvar` | MVAr | Reactive power arriving at "to" end |

---



## 3. The CIGRE MV Benchmark Network

Gridifix uses the **CIGRE (Conseil International des Grands Réseaux Électriques) MV distribution network** as its benchmark test case. This network is a standard reference model for evaluating distribution system analysis tools.

### 3.1 Network Topology

The CIGRE MV network consists of:

- **14 Load Buses** (labeled 0 through 13): Each connected to one or more loads.
- **1 Slack Bus** (typically bus 0): The reference bus for power flow.
- **15 Lines** (labeled 0 through 14): Overhead lines or cables connecting the buses, forming a radial/meshed structure.

```
Simplified CIGRE MV Layout (Linux 7-Bus Equivalent Core):

Slack Bus (Bus 0)
    ├── Line 0 → Bus 1 ──┬── Line 4 → Bus 5
    │                     └── Line 5 → Bus 6
    ├── Line 1 → Bus 2 ──┬── Line 6 → Bus 7
    │                     └── Line 7 → Bus 8
    ├── Line 2 → Bus 3 ──┬── Line 8 → Bus 9
    │                     └── Line 9 → Bus 10
    └── ... (extended to 14 buses + lines in full model)
```

> **Note:** The actual CIGRE MV network as implemented in `pandapower` has 15 buses and 15 lines with specific line lengths, types, and load values. Gridifix uses `pandapower.networks.create_cigre_network_mv()` to instantiate this topology.

### 3.2 Network Creation in pandapower

```python
from pandapower.networks import create_cigre_network_mv
import pandapower as pp

# Instantiate the CIGRE MV benchmark network
net = create_cigre_network_mv()

# Run power flow analysis
pp.runpp(net)

# Check if power flow converged
if net.converged:
    bus_vm = net.res_bus.vm_pu      # Voltage magnitudes
    line_p = net.res_line.p_from_mw # Line power flows
```

### 3.3 Network Elements in pandapower

| Element | DataFrame Key | Purpose |
|---------|---------------|---------|
| Buses | `net.bus` | Network nodes; each has a unique `bus_id` |
| Lines | `net.line` | Branches connecting buses; `from_bus`, `to_bus`, `length_km`, `std_type` |
| Loads | `net.load` | Consumer demand; `p_mw` (active), `q_mvar` (reactive), connected to a bus |
| Static Generators | `net.sgen` | Distributed generation; `p_mw`, `q_mvar`, connected to a bus |
| Trafos | `net.trafo` | Transformers (if any in the model) |

---



## 4. pandapower Overview

`pandapower` is an open-source Python tool for **power system analysis and optimization**, built on top of `pandas` and `numpy`. It provides:

### 4.1 Core Capabilities

| Feature | Description |
|---------|-------------|
| Power Flow | Newton-Raphson & DC approximations (`pp.runpp`) |
| Short-Circuit | IEC 60909 and IEC 60909 methods |
| State Estimation | Weighted Least Squares (WLS) |
| Optimal Power Flow | Linear and non-linear OPF |
| Time Series | Multi-step simulation with `pandapower.control` |

### 4.2 Data Structures

pandapower uses **pandas DataFrames** to store all network data:

```python
# Network bus data
net.bus            # Columns: bus_id, vn_kv, in_service, etc.
net.load           # Columns: load_id, bus, p_mw, q_mvar, in_service
net.line           # Columns: line_id, from_bus, to_bus, length_km, r_ohm_per_km, x_ohm_per_km, etc.
net.sgen           # Columns: sgen_id, bus, p_mw, q_mvar, type
net.res_bus        # Results: vm_pu, va_degree (after pp.runpp)
net.res_line       # Results: p_from_mw, q_from_mvar, p_to_mw, q_to_mvar, etc.
```

### 4.3 Power Flow Results

After calling `pp.runpp(net)`:

- `net.converged` → `True` if the solver found a valid solution
- `net.res_bus` → DataFrame with voltage magnitudes (`vm_pu`) and angles (`va_degree`) for each bus
- `net.res_line` → DataFrame with power flows (`p_from_mw`, `q_from_mvar`, `p_to_mw`, `q_to_mvar`) for each line

---



## 5. Newton-Raphson vs. Distribution Linear Power Flow

### 5.1 Newton-Raphson (NR) Method

The standard AC power flow solver used by pandapower. It iteratively solves the **non-linear** power flow equations:

```
P_i = V_i Σ_j |Y_ij| V_j cos(θ_i - θ_j - φ_ij)
Q_i = V_i Σ_j |Y_ij| V_j sin(θ_i - θ_ij - φ_ij)
```

**Pros:** Accurate for all voltage levels and network conditions.
**Cons:** Computationally expensive; convergence not guaranteed for ill-conditioned systems.

### 5.2 Distribution Linear Power Flow (DLPF)

Gridifix's custom solver replaces NR with a **linearized approximation** suitable for distribution networks where:

- Voltage angles are small (near the slack bus reference)
- The R/X ratio is high
- Voltage magnitudes vary within ±5–10% of nominal

The DLPF formulation approximates:

```
ΔV ≈ R·P + X·Q      (for radial feeders with small angle approximations)
```

**Pros:** 10x+ faster than NR; deterministic convergence; suitable for online/real-time applications.
**Cons:** Accuracy degrades for heavily loaded or meshed networks with large voltage deviations.

**Gridifix's DLPF implementation:** See [`src/dlpf-solver/`](https://github.com/Divyesh-Kamalanaban/gridifix/tree/main/src/dlpf-solver) for the native Python solver.

---



## 6. Summary

| Concept | Role in Gridifix |
|---------|-----------------|
| MV Distribution Network | The physical system being monitored |
| CIGRE MV Benchmark | The standard test topology used for all simulations |
| pandapower | Simulation engine generating training data and real-time measurements |
| Power Flow | Provides the voltage/angle/power-flow targets for DNN training |
| DLPF Solver | High-speed solver incorporated into the pipeline (pre-training) |
| Newton-Raphson | Accurate reference solver used by pandapower by default |

---

**Next:** [Pipeline Overview](Pipeline-Overview.md) — Learn how Gridifix combines these concepts into a complete ML-powered fault detection pipeline.