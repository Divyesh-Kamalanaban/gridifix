# main.py (Minimal Version: Neural Network vs. Newton-Raphson Comparison)

import pandas as pd
import tensorflow as tf
import joblib
import sys
import pandapower as pp
import numpy as np
import time
from pandapower.networks import create_cigre_network_mv
from sklearn.metrics import mean_absolute_error

# --- 1. Model and Scaler Paths ---
NN_MODEL_PATH = 'power_flow_nn_model.h5'
INPUT_SCALER_PATH = 'input_scaler.pkl'
OUTPUT_SCALER_PATH = 'output_scaler.pkl'

print("\n--- Loading required models and scalers ---")

# --- 2. Load Models and Scalers (No Error Handling as requested) ---
loaded_nn_model = tf.keras.models.load_model(NN_MODEL_PATH)
loaded_input_scaler = joblib.load(INPUT_SCALER_PATH)
loaded_output_scaler = joblib.load(OUTPUT_SCALER_PATH)
print("Successfully loaded power flow NN model and its scalers.")

# --- 3. Dynamically Determine NN Input and Output Columns ---
print("\n--- Identifying NN training column names ---")
temp_inputs_df_for_cols = pd.read_csv("combined_power_flow_inputs.csv")
temp_outputs_df_for_cols = pd.read_csv("combined_power_flow_outputs.csv")

constant_input_cols_for_nn = [col for col in temp_inputs_df_for_cols.columns if
                              temp_inputs_df_for_cols[col].nunique() == 1]
original_nn_input_columns = temp_inputs_df_for_cols.drop(
    columns=constant_input_cols_for_nn).columns.tolist() if constant_input_cols_for_nn else temp_inputs_df_for_cols.columns.tolist()

constant_output_cols_for_nn = [col for col in temp_outputs_df_for_cols.columns if
                               temp_outputs_df_for_cols[col].nunique() == 1]
original_nn_output_columns = temp_outputs_df_for_cols.drop(
    columns=constant_output_cols_for_nn).columns.tolist() if constant_output_cols_for_nn else temp_outputs_df_for_cols.columns.tolist()

print(f"Dynamically determined NN input columns count: {len(original_nn_input_columns)}")
print(f"Dynamically determined NN output columns count: {len(original_nn_output_columns)}")
print("NN training column names determined successfully.")

# --- Main execution block for NN vs. NR Comparison ---
if __name__ == '__main__':
    print("\n--- Running Neural Network vs. Newton-Raphson Power Flow Comparison ---")

    # 1. Setup a test network for comparison (a fresh CIGRE MV network)
    test_net_comparison = create_cigre_network_mv()

    # 2. Measure Newton-Raphson (NR) Solver Time and get its results
    print("\nMeasuring Newton-Raphson solver performance...")
    nr_start_time = time.time()
    pp.runpp(test_net_comparison)  # Run power flow using NR solver
    nr_end_time = time.time()
    nr_time = nr_end_time - nr_start_time
    print(f"Newton-Raphson Solver Time: {nr_time:.6f} seconds")

    # Extract NR results for comparison (ground truth for NN accuracy)
    nr_actual_measurements_dict = {col: 0.0 for col in original_nn_output_columns}
    nr_bus_results = test_net_comparison.res_bus
    nr_line_results = test_net_comparison.res_line

    for id in test_net_comparison.bus.index.to_numpy():
        if f'bus_{id}_vm_pu' in original_nn_output_columns:
            nr_actual_measurements_dict[f'bus_{id}_vm_pu'] = nr_bus_results.at[id, 'vm_pu']
        if f'bus_{id}_va_degree' in original_nn_output_columns:
            nr_actual_measurements_dict[f'bus_{id}_va_degree'] = nr_bus_results.at[id, 'va_degree']
    for id in test_net_comparison.line.index.to_numpy():
        if id in nr_line_results.index:
            nr_actual_measurements_dict[f'line_{id}_p_from_mw'] = nr_line_results.at[id, 'p_from_mw']
            nr_actual_measurements_dict[f'line_{id}_q_from_mvar'] = nr_line_results.at[id, 'q_from_mvar']
            nr_actual_measurements_dict[f'line_{id}_p_to_mw'] = nr_line_results.at[id, 'p_to_mw']
            nr_actual_measurements_dict[f'line_{id}_q_to_mvar'] = nr_line_results.at[id, 'q_to_mvar']
    nr_actual_outputs_df = pd.DataFrame([nr_actual_measurements_dict], columns=original_nn_output_columns)

    # 3. Measure NN Prediction Time and get its results
    print("\nMeasuring Neural Network prediction performance...")
    nn_pred_start_time = time.time()

    # Replicate NN input extraction from a typical scenario
    current_inputs_dict_nn = {}
    for idx, load_rows in test_net_comparison.load.iterrows():
        current_inputs_dict_nn[f'load_{idx}_p_mw'] = load_rows['p_mw']
        current_inputs_dict_nn[f'load_{idx}_q_mvar'] = load_rows['q_mvar']
    for col in original_nn_input_columns:
        if 'sgen_bus_' in col and col not in current_inputs_dict_nn:
            current_inputs_dict_nn[col] = 0.0
    sgen_power_at_bus_nn = test_net_comparison.sgen.groupby('bus')[['p_mw', 'q_mvar']].sum().to_dict('index')
    for bus_id, values in sgen_power_at_bus_nn.items():
        current_inputs_dict_nn[f'sgen_bus_{bus_id}_p_mw'] = values['p_mw']
        current_inputs_dict_nn[f'sgen_bus_{bus_id}_q_mvar'] = values['q_mvar']
    nn_input_df_row_comparison = pd.DataFrame([current_inputs_dict_nn], columns=original_nn_input_columns).fillna(0)

    # Perform NN prediction
    scaled_nn_inputs_comparison = loaded_input_scaler.transform(nn_input_df_row_comparison)
    nn_predicted_outputs_scaled_comparison = loaded_nn_model.predict(scaled_nn_inputs_comparison, verbose=0)

    # FIX: Convert the inverse_transform result back to a DataFrame
    nn_predicted_outputs_original_np = loaded_output_scaler.inverse_transform(nn_predicted_outputs_scaled_comparison)
    nn_predicted_outputs_original_df_comparison = pd.DataFrame(nn_predicted_outputs_original_np,
                                                               columns=original_nn_output_columns)

    nn_pred_end_time = time.time()
    nn_time = nn_pred_end_time - nn_pred_start_time
    print(f"Neural Network Prediction Time: {nn_time:.6f} seconds")

    # 4. Calculate Speed Reduction
    speed_reduction_percent = ((nr_time - nn_time) / nr_time) * 100
    print(f"Speed Reduction (NN vs. NR): {speed_reduction_percent:.2f}%")

    # 5. Calculate Accuracy (MAE) of NN's Power Flow Prediction vs. NR (Overall)
    overall_mae = mean_absolute_error(nr_actual_outputs_df, nn_predicted_outputs_original_df_comparison)
    print(f"\nOverall Mean Absolute Error (MAE) of NN Power Flow Prediction vs. NR: {overall_mae:.9f}")

    # 6. Calculate MAE for each individual metric
    print("\n--- Mean Absolute Error (MAE) for Each Output Metric ---")
    for col in original_nn_output_columns:
        mae_per_metric = mean_absolute_error(nr_actual_outputs_df[col],
                                             nn_predicted_outputs_original_df_comparison[col])
        print(f"  MAE for {col}: {mae_per_metric:.9f}")

    print("\n--- Neural Network vs. Newton-Raphson Comparison Complete ---")
