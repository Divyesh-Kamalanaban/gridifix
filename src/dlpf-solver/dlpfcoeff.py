import numpy as np
import pandapower as pp

def get_dlpf_coeff(net):
    # admittance matrix - ybus, from bus admittance - yf, to bus admittance - yt
    [y_bus, yf_bus, yt_bus] = pp.makeYbus_pypower(baseMVA=net["_ppc"]['baseMVA'], bus=net["_ppc"]['bus'],
                                                  branch=net["_ppc"]['branch'])
    # dense form of y_bus
    y_bus_dense = y_bus.todense()
    # conductance - g_mat - real part of y_bus
    g_mat = np.asarray(y_bus_dense.real)
    # susceptance - b_mat - imag part of y_bus
    b_mat = np.asarray(y_bus_dense.imag)

    # getting indexes of slack, pv and pq buses through pypower internal func
    [sl_bus_raw, pv_bus_raw, pq_bus_raw] = pp.bustypes(bus=net["_ppc"]['bus'], gen=net["_ppc"]['gen'])

    # Get the total number of buses in the network from net.bus
    num_total_buses = net.bus.shape[0]

    # Filter the raw bus type arrays
    sl_bus = np.array([idx for idx in sl_bus_raw if idx < num_total_buses], dtype=int)
    pv_bus = np.array([idx for idx in pv_bus_raw if idx < num_total_buses], dtype=int)
    # Corrected PQ bus filtering: should be all non-slack and non-PV buses
    pq_bus = np.array([i for i in range(num_total_buses) if i not in sl_bus and i not in pv_bus], dtype=int)

    print(f"\n--- Bus Types for this Run ---")
    print(f"Slack (R) Buses: {sl_bus}, Dtype: {sl_bus.dtype}")
    print(f"PV (S) Buses: {pv_bus}, Dtype: {pv_bus.dtype}")
    print(f"PQ (L) Buses: {pq_bus}, Dtype: {pq_bus.dtype}")
    print(f"Total Buses (from net.bus.shape[0]): {num_total_buses}")

    # getting conductance matrices:
    # L = pq_bus, S = pv_bus, R = sl_bus
    # Using np.zeros for empty cases to ensure correct 0-dimensions for np.block/hstack/vstack
    g_lr = g_mat[np.ix_(pq_bus, sl_bus)] if pq_bus.size > 0 and sl_bus.size > 0 else np.zeros(
        (pq_bus.size, sl_bus.size))
    g_ls = g_mat[np.ix_(pq_bus, pv_bus)] if pq_bus.size > 0 and pv_bus.size > 0 else np.zeros(
        (pq_bus.size, pv_bus.size))
    g_ll = g_mat[np.ix_(pq_bus, pq_bus)] if pq_bus.size > 0 else np.zeros((pq_bus.size, pq_bus.size))

    g_rl = g_mat[np.ix_(sl_bus, pq_bus)] if sl_bus.size > 0 and pq_bus.size > 0 else np.zeros(
        (sl_bus.size, pq_bus.size))
    g_rs = g_mat[np.ix_(sl_bus, pv_bus)] if sl_bus.size > 0 and pv_bus.size > 0 else np.zeros(
        (sl_bus.size, pv_bus.size))
    g_rr = g_mat[np.ix_(sl_bus, sl_bus)] if sl_bus.size > 0 else np.zeros((sl_bus.size, sl_bus.size))

    g_sl = g_mat[np.ix_(pv_bus, pq_bus)] if pv_bus.size > 0 and pq_bus.size > 0 else np.zeros(
        (pv_bus.size, pq_bus.size))
    g_sr = g_mat[np.ix_(pv_bus, sl_bus)] if pv_bus.size > 0 and sl_bus.size > 0 else np.zeros(
        (pv_bus.size, sl_bus.size))
    g_ss = g_mat[np.ix_(pv_bus, pv_bus)] if pv_bus.size > 0 else np.zeros((pv_bus.size, pv_bus.size))

    # getting susceptance matrices with LSR notation:
    # L = pq_bus, S = pv_bus, R = sl_bus
    b_lr = b_mat[np.ix_(pq_bus, sl_bus)] if pq_bus.size > 0 and sl_bus.size > 0 else np.zeros(
        (pq_bus.size, sl_bus.size))
    b_ls = b_mat[np.ix_(pq_bus, pv_bus)] if pq_bus.size > 0 and pv_bus.size > 0 else np.zeros(
        (pq_bus.size, pv_bus.size))
    b_ll = b_mat[np.ix_(pq_bus, pq_bus)] if pq_bus.size > 0 else np.zeros((pq_bus.size, pq_bus.size))

    b_rl = b_mat[np.ix_(sl_bus, pq_bus)] if sl_bus.size > 0 and pq_bus.size > 0 else np.zeros(
        (sl_bus.size, pq_bus.size))
    b_rs = b_mat[np.ix_(sl_bus, pv_bus)] if sl_bus.size > 0 and pv_bus.size > 0 else np.zeros(
        (sl_bus.size, pv_bus.size))
    b_rr = b_mat[np.ix_(sl_bus, sl_bus)] if sl_bus.size > 0 else np.zeros((sl_bus.size, sl_bus.size))

    b_sl = b_mat[np.ix_(pv_bus, pq_bus)] if pv_bus.size > 0 and pq_bus.size > 0 else np.zeros(
        (pv_bus.size, pq_bus.size))
    b_sr = b_mat[np.ix_(pv_bus, sl_bus)] if pv_bus.size > 0 and sl_bus.size > 0 else np.zeros(
        (pv_bus.size, sl_bus.size))
    b_ss = b_mat[np.ix_(pv_bus, pv_bus)] if pv_bus.size > 0 else np.zeros((pv_bus.size, pv_bus.size))

    # Combined indices for theta_S and theta_L.
    # The order matters for the blocks: PV (S) then PQ (L)
    theta_unknown_indices = np.concatenate((pv_bus, pq_bus))
    # Indices for V_L.
    v_unknown_indices = pq_bus

    # H: Size (pv_bus.size + pq_bus.size) x (pv_bus.size + pq_bus.size)
    H_size = pv_bus.size + pq_bus.size
    H = np.zeros((H_size, H_size))

    if pv_bus.size > 0:
        H[:pv_bus.size, :pv_bus.size] = b_ss  # Top-left block (PV-PV)
        if pq_bus.size > 0:
            H[:pv_bus.size, pv_bus.size:] = b_sl  # Top-right block (PV-PQ)

    if pq_bus.size > 0:
        if pv_bus.size > 0:
            H[pv_bus.size:, :pv_bus.size] = b_ls  # Bottom-left block (PQ-PV)
        H[pv_bus.size:, pv_bus.size:] = b_ll  # Bottom-right block (PQ-PQ)

    H = -H  # Apply the negative sign to the full H matrix

    # N: Size (pv_bus.size + pq_bus.size) x pq_bus.size
    N_rows = pv_bus.size + pq_bus.size
    N_cols = pq_bus.size
    N = np.zeros((N_rows, N_cols))
    if pv_bus.size > 0 and pq_bus.size > 0:  # g_sl
        N[:pv_bus.size, :] = g_sl
    if pq_bus.size > 0:  # g_ll
        N[pv_bus.size:, :] = g_ll

    # M: Size pq_bus.size x (pv_bus.size + pq_bus.size)
    M_rows = pq_bus.size
    M_cols = pv_bus.size + pq_bus.size
    M = np.zeros((M_rows, M_cols))
    if pq_bus.size > 0 and pv_bus.size > 0:  # g_ls
        M[:, :pv_bus.size] = g_ls
    if pq_bus.size > 0:  # g_ll (again for M)
        M[:, pv_bus.size:] = g_ll

    M = -M  # Apply the negative sign to the full M matrix

    # L: Size pq_bus.size x pq_bus.size
    L = -b_ll  # This one is straightforward as b_ll will always have correct size based on pq_bus.size

    # Robust inverse handling for empty or singular matrices
    L_inv = np.linalg.inv(L) if L.size > 0 and np.linalg.det(L) != 0 else np.zeros_like(L)
    H_inv = np.linalg.inv(H) if H.size > 0 and np.linalg.det(H) != 0 else np.zeros_like(H)

    # --- Calculating Decoupled System Matrices (tilde_H, tilde_L) and their Inverses ---
    print("\n--- Calculating Decoupled System Matrices (tilde_H, tilde_L) ---")

    # Ensure matrix operations are robust to empty matrices
    # Check for empty N @ L_inv @ M before subtraction
    term_H_subtract = N @ L_inv @ M if N.size > 0 and L_inv.size > 0 and M.size > 0 else np.zeros(H.shape)
    tilde_H = H - term_H_subtract
    tilde_H_inv = np.linalg.inv(tilde_H) if tilde_H.size > 0 and np.linalg.det(tilde_H) != 0 else np.zeros_like(tilde_H)

    # Check for empty M @ H_inv @ N before subtraction
    term_L_subtract = M @ H_inv @ N if M.size > 0 and H_inv.size > 0 and N.size > 0 else np.zeros(L.shape)
    tilde_L = L - term_L_subtract
    tilde_L_inv = np.linalg.inv(tilde_L) if tilde_L.size > 0 and np.linalg.det(tilde_L) != 0 else np.zeros_like(tilde_L)

    print(f"Shape of tilde_H: {tilde_H.shape}")
    print(f"Shape of tilde_H_inv: {tilde_H_inv.shape}")
    print(f"Shape of tilde_L: {tilde_L.shape}")
    print(f"Shape of tilde_L_inv: {tilde_L_inv.shape}")

    # --- Calculate Solution Coefficient Matrices (A, B, C, D) ---
    A = tilde_H_inv

    # B requires multiplication with potentially empty N and tilde_L_inv
    if N.size > 0 and tilde_L_inv.size > 0 and N.shape[1] == tilde_L_inv.shape[0]:
        B = -tilde_H_inv @ N @ tilde_L_inv
    else:
        # If N or tilde_L_inv are empty, B should be sized correctly for the output
        # It maps Q_L (pq_bus.size) to theta_SL_dlpf (pv_bus.size + pq_bus.size)
        B = np.zeros((pv_bus.size + pq_bus.size, pq_bus.size))

    # C requires multiplication with potentially empty M and tilde_H_inv
    if M.size > 0 and tilde_H_inv.size > 0 and M.shape[1] == tilde_H_inv.shape[0]:
        C = -tilde_L_inv @ M @ tilde_H_inv
    else:
        # If M or tilde_H_inv are empty, C should be sized correctly for the output
        # It maps P_SL (pv_bus.size + pq_bus.size) to V_L_dlpf (pq_bus.size)
        C = np.zeros((pq_bus.size, pv_bus.size + pq_bus.size))

    D = tilde_L_inv

    # Get slack bus (reference) voltage and angle (in radians) from the current net results
    V_R_pu = net.res_bus.vm_pu.values[sl_bus[0]] if sl_bus.size > 0 else 1.0
    theta_R_rad = np.deg2rad(net.res_bus.va_degree.values[sl_bus[0]]) if sl_bus.size > 0 else 0.0

    # Return only the coefficients and bus type/reference data
    return A, B, C, D, pv_bus, pq_bus, sl_bus, num_total_buses
