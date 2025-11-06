import numpy as np

def solve_dlpf(net, A, B, C, D, pq_bus, pv_bus, sl_bus, num_total_buses):


    # --- Power Injections for DLPF Solution ---
    all_p_res = net.res_bus.p_mw.values  # active power result values
    all_q_res = net.res_bus.q_mvar.values  # reactive power result values

    # --- DEBUG: Inside solve_dlpf for Power Injections ---
    print("\n--- DEBUG: Inside solve_dlpf for Power Injections ---")
    print(f"pq_bus received: {pq_bus}, size: {pq_bus.size}")
    print(f"pv_bus received: {pv_bus}, size: {pv_bus.size}")
    print(f"all_q_res length: {len(all_q_res)}")
    # Print only the first few elements to avoid clutter for large arrays
    print(f"all_q_res (first 5 elements): {all_q_res[:min(5, len(all_q_res))]}")
    print("-------------------------------------------------------")

    # injections for PV and PQ buses - reshaped to 1 column 2-d arrays for matrix multiplication
    P_S = all_p_res[pv_bus].reshape(-1, 1) if pv_bus.size > 0 else np.zeros((0, 1))
    P_L = all_p_res[pq_bus].reshape(-1, 1) if pq_bus.size > 0 else np.zeros((0, 1))
    Q_L = all_q_res[pq_bus].reshape(-1, 1) if pq_bus.size > 0 else np.zeros((0, 1))
    print(f"Shape of Q_L inside solve_dlpf: {Q_L.shape}")
    P_SL = np.concatenate((P_S, P_L)) if (P_S.size > 0 or P_L.size > 0) else np.zeros((0, 1))

    # Get slack bus (reference) voltage and angle (in radians)
    V_R_pu = net.res_bus.vm_pu.values[sl_bus[0]] if sl_bus.size > 0 else 1.0  # ext grid - 1.03
    theta_R_rad = np.deg2rad(net.res_bus.va_degree.values[sl_bus[0]]) if sl_bus.size > 0 else 0.0  # ext grid - 0

    print(f"\n--- Power Injections for DLPF Solution ---")
    print(f"Slack Bus Voltage (V_R_pu): {V_R_pu:.4f} pu, Angle (theta_R_rad): {theta_R_rad:.4f} rad")
    # --- DLPF SOLUTION: Solve for Unknown Voltage Phase Angles (theta_[S;L]) and Voltage Magnitudes (V_L) ---

    # Removed duplicate calculations
    # Calculate A @ P_SL. Initialize with NaNs to indicate if calculation failed.
    term_A_P_SL = np.full((A.shape[0], 1), np.nan)
    if P_SL.shape[0] > 0 and A.shape[1] == P_SL.shape[0]:  # If P_SL is not empty AND dimensions match
        term_A_P_SL = A @ P_SL
    elif P_SL.shape[0] == 0 and A.shape[1] == 0:  # If both P_SL and A's relevant dimension are 0, valid empty product.
        term_A_P_SL = np.zeros((A.shape[0], 1))
    else:  # Dimension mismatch for non-empty arrays
        print(
            f"ERROR: A column count ({A.shape[1]}) mismatches P_SL row count ({P_SL.shape[0]}) for A@P_SL. Result will be NaN.")

    # Calculate B @ Q_L. Initialize with NaNs.
    term_B_Q_L = np.full((B.shape[0], 1), np.nan)
    if Q_L.shape[0] > 0 and B.shape[1] == Q_L.shape[0]:  # If Q_L is not empty AND dimensions match
        term_B_Q_L = B @ Q_L
    elif Q_L.shape[0] == 0 and B.shape[1] == 0:  # If both Q_L and B's relevant dimension are 0, valid empty product.
        term_B_Q_L = np.zeros((B.shape[0], 1))
    else:  # Dimension mismatch for non-empty arrays
        print(
            f"ERROR: B column count ({B.shape[1]}) mismatches Q_L row count ({Q_L.shape[0]}) for B@Q_L. Result will be NaN.")

    # Sum the terms. If both terms are NaN, the result is NaN.
    theta_SL_dlpf = term_A_P_SL + term_B_Q_L
    if np.isnan(term_A_P_SL).all() and np.isnan(term_B_Q_L).all():
        theta_SL_dlpf = np.full((A.shape[0], 1), np.nan)
    # If there are no PV or PQ buses in total, theta_SL_dlpf should be a 0x1 array.
    elif (pv_bus.size + pq_bus.size) == 0:
        theta_SL_dlpf = np.zeros((0, 1))
    print(f"Shape of calculated theta_SL_dlpf: {theta_SL_dlpf.shape}")
    print(f"Calculated theta_SL_dlpf values:\n{theta_SL_dlpf}")

    # Calculate C @ P_SL. Initialize with NaNs.
    term_C_P_SL = np.full((C.shape[0], 1), np.nan)
    if P_SL.shape[0] > 0 and C.shape[1] == P_SL.shape[0]:
        term_C_P_SL = C @ P_SL
    elif P_SL.shape[0] == 0 and C.shape[1] == 0:
        term_C_P_SL = np.zeros((C.shape[0], 1))
    else:
        print(
            f"ERROR: C column count ({C.shape[1]}) mismatches P_SL row count ({P_SL.shape[0]}) for C@P_SL. Result will be NaN.")

    # Calculate D @ Q_L. Initialize with NaNs.
    term_D_Q_L = np.full((D.shape[0], 1), np.nan)
    if Q_L.shape[0] > 0 and D.shape[1] == Q_L.shape[0]:
        term_D_Q_L = D @ Q_L
    elif Q_L.shape[0] == 0 and D.shape[1] == 0:
        term_D_Q_L = np.zeros((D.shape[0], 1))
    else:
        print(f"ERROR: D column count ({D.shape[1]}) mismatches Q_L row count ({Q_L.shape[0]}) for D@Q_L. Result will be NaN.")

    # Sum the terms. If both terms are NaN, the result is NaN.
    V_L_dlpf = term_C_P_SL + term_D_Q_L
    if np.isnan(term_C_P_SL).all() and np.isnan(term_D_Q_L).all():
        V_L_dlpf = np.full((C.shape[0], 1), np.nan)
    # If there are no PQ buses, V_L_dlpf_deviation should be a 0x1 array.
    elif pq_bus.size == 0:
        V_L_dlpf = np.zeros((0, 1))

    print(f"Shape of calculated V_L_dlpf: {V_L_dlpf.shape}")
    print(f"Calculated V_L_dlpf values:\n{V_L_dlpf}")

    # --- DLPF Solution assembly ---
    total_buses = num_total_buses
    # Initialized with 0.0 and 1.0 as requested
    dlpf_theta_rad = np.full(total_buses, 0.0)
    dlpf_vm_pu = np.full(total_buses, 1.0)

    if sl_bus.size > 0:
        dlpf_theta_rad[sl_bus[0]] = theta_R_rad
        dlpf_vm_pu[sl_bus[0]] = V_R_pu

    # Separate theta_SL_dlpf into theta_S and theta_L
    theta_S_dlpf = theta_SL_dlpf[0:pv_bus.size]
    theta_L_dlpf = theta_SL_dlpf[pv_bus.size:]

    # REINTRODUCED ROBUST ASSIGNMENT BLOCKS AND DEBUG PRINTS
    if theta_SL_dlpf is not None and theta_SL_dlpf.size > 0:
        # Assigining magnitudes
        if pv_bus.size > 0 and theta_S_dlpf.size > 0:
            dlpf_theta_rad[pv_bus] = theta_S_dlpf.flatten()  # Store PV angles
            dlpf_vm_pu[pv_bus] = net.res_bus.vm_pu.values[pv_bus].flatten()  # Use initial VM for PV buses
        if pq_bus.size > 0 and theta_L_dlpf.size > 0:
            dlpf_theta_rad[pq_bus] = theta_L_dlpf.flatten()  # Store PQ angles - REINTRODUCED THIS CRITICAL LINE
            if V_L_dlpf is not None and V_L_dlpf.size > 0:
                # Store PQ magnitudes - CORRECTED to add 1.0
                dlpf_vm_pu[pq_bus] = 1 + V_L_dlpf.flatten()
        else:
            print("WARNING: PQ bus size is 0 or theta_L_dlpf is empty. PQ angles/magnitudes not assigned.")
    else:
        print("\nWARNING: theta_SL_dlpf not calculated or is empty. Angles and some magnitudes may remain 0.0 or 1.0.")

    print("\n--- DLPF Solved Voltages (All Buses) ---")
    print(f"DLPF Voltage Magnitudes (pu):\n{dlpf_vm_pu}")
    print(f"DLPF Voltage Angles (rad):\n{dlpf_theta_rad}")
    print(f"DLPF Voltage Angles (deg):\n{np.rad2deg(dlpf_theta_rad)}")



    return dlpf_vm_pu,dlpf_theta_rad
