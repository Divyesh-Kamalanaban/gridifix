import numpy as np

#produces error analysis results by comparing dlpf results with powerflow analysis results; useful for error stats.
def error_analysis_dlpf(net, dlpf_vm_pu, dlpf_theta_rad):
    # For comparison, get actual results from pandapower's power flow
    actual_vm_pu = net.res_bus.vm_pu.values
    actual_va_deg = net.res_bus.va_degree.values
    actual_va_rad = np.deg2rad(actual_va_deg)

    print("\n--- Actual Pandapower Power Flow Results (for comparison) ---")
    print(f"Actual Voltage Magnitudes (pu):\n{actual_vm_pu}")
    print(f"Actual Voltage Angles (rad):\n{actual_va_rad}")
    print(f"Actual Voltage Angles (deg):\n{actual_va_deg}")

    # Calculate errors
    vm_error = np.abs(dlpf_vm_pu - actual_vm_pu)
    va_error = np.abs(dlpf_theta_rad - actual_va_rad)

    print("\n--- Error Analysis (DLPF vs Pandapower PF) ---")
    print(f"Max Absolute Error in Voltage Magnitude (pu): {np.nanmax(vm_error):.6f}")
    print(f"Mean Absolute Error in Voltage Magnitude (pu): {np.nanmean(vm_error):.6f}")
    print(f"Max Absolute Error in Voltage Angle (rad): {np.nanmax(va_error):.6f}")
    print(f"Mean Absolute Error in Voltage Angle (rad): {np.nanmean(va_error):.6f}")
