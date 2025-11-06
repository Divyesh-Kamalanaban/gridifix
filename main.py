# main.py (Updated with scaler feature name debug prints)

import initcigre
import dlpfcoeff
import solvedlpf
import erroranalysisdlpf
import dermodelling
import samplegenerator  # Import your module containing generate_samples
import pandas as pd
import tensorflow as tf
import joblib
import sys  # For sys.exit()
import ydf
import pandapower as pp
import numpy as np
from pandapower.networks import create_cigre_network_mv

# try excerpt loading the saved NN model (h5) and the scalers (pkl)
try:
    loaded_nn_model = tf.keras.models.load_model('power_flow_nn_model.h5')
    loaded_input_scaler = joblib.load('input_scaler.pkl')
    loaded_output_scaler = joblib.load('output_scaler.pkl')
    fault_detection_model = ydf.load_model('fault_detection_model/content/fault_detection_model')
    fault_locate_model = ydf.load_model('fault_locate_model/content/fault_locate_model')
    print("Successfully loaded power flow NN model, its scalers and the Random Forest Regressors in main script.")
    print("\n--- DEBUG: Fault Localization Model Summary (after loading) ---")
    print(f"--- DEBUG: Type of fault_locate_model.label_classes: {fault_locate_model.label_classes()}")

except FileNotFoundError as e:
    print(f"FATAL ERROR: Required model or scaler file not found: {e}")
    print(
        "Please ensure 'power_flow_nn_model.h5', 'input_scaler.pkl', and 'output_scaler.pkl' are in the same directory.")
    sys.exit(1)
except Exception as e:
    print( f"FATAL ERROR: An unexpected error occurred while loading NN model ,its scalers or the Random Forest Regressors: {e}")
    sys.exit(1)

#defining the nn input and output columns:
temp_inputs_df_for_cols = pd.read_csv("combined_power_flow_inputs.csv")
temp_outputs_df_for_cols = pd.read_csv("combined_power_flow_outputs.csv")

constant_input_cols_for_nn = [col for col in temp_inputs_df_for_cols.columns if temp_inputs_df_for_cols[col].nunique() == 1]
if constant_input_cols_for_nn:
    original_nn_input_columns = temp_inputs_df_for_cols.drop(columns=constant_input_cols_for_nn).columns.tolist()
else:
    original_nn_input_columns = temp_inputs_df_for_cols.columns.tolist()

constant_output_cols_for_nn = [col for col in temp_outputs_df_for_cols.columns if temp_outputs_df_for_cols[col].nunique() == 1]
if constant_output_cols_for_nn:
    original_nn_output_columns = temp_outputs_df_for_cols.drop(columns=constant_output_cols_for_nn).columns.tolist()
else:
    original_nn_output_columns = temp_outputs_df_for_cols.columns.tolist()

print(f"Dynamically determined NN input columns count: {len(original_nn_input_columns)}")
print(f"Dynamically determined NN output columns count: {len(original_nn_output_columns)}")
print("NN training column names determined successfully.")

def predict_fault_status(current_net: pp.pandapowerNet) -> dict:
    if not current_net.converged:
        return {"status": "Power Flow Not Converged",
                "message": "Cannot perform inference: Power flow did not converge for the provided network state."}

    #copy the exact logic from samplegenerator.py
    current_inputs_dict = {}

    # Loads
    for idx, load_rows in current_net.load.iterrows():
        current_inputs_dict[f'load_{idx}_p_mw'] = load_rows['p_mw']
        current_inputs_dict[f'load_{idx}_q_mvar'] = load_rows['q_mvar']

    # Sgens (need to iterate through existing sgens to get their current values)
    # The 'sgen_bus_X_p_mw' columns expect the total sgen at a bus.
    # We need to aggregate if multiple sgens are at one bus.
    # A safer way is to iterate through original_nn_input_columns to build this dataframe
    # and fill missing with 0, as done in samplegenerator.py

    #default fallback as 0 for sgens init by default.
    for col in original_nn_input_columns:
        if 'sgen_bus_' in col and col not in current_inputs_dict:
            current_inputs_dict[col] = 0.0

    #getting the 2 cols for sgen power for each bus and aggregating it.
    sgen_power_at_bus = current_net.sgen.groupby('bus')[['p_mw', 'q_mvar']].sum().to_dict('index')
    for bus_id, values in sgen_power_at_bus.items():
        current_inputs_dict[f'sgen_bus_{bus_id}_p_mw'] = values['p_mw']
        current_inputs_dict[f'sgen_bus_{bus_id}_q_mvar'] = values['q_mvar']

    #compiling the input dict into a df with appropriate cols
    nn_input_df_row = pd.DataFrame([current_inputs_dict], columns=original_nn_input_columns).fillna(0)

    #scaling it with the imported scalers for the neural network.
    scaled_nn_inputs = loaded_input_scaler.transform(nn_input_df_row)
    nn_predicted_outputs_scaled = loaded_nn_model.predict(scaled_nn_inputs, verbose=0)

    # Convert NN output back to DataFrame for inverse scaling with column names
    nn_predicted_outputs_scaled_df = pd.DataFrame(nn_predicted_outputs_scaled, columns=original_nn_output_columns)
    nn_predicted_outputs_original = loaded_output_scaler.inverse_transform(nn_predicted_outputs_scaled_df)
    nn_predicted_outputs_original_df = pd.DataFrame(nn_predicted_outputs_original,columns=original_nn_output_columns)

    #extracting outputs from NN predictions to feed into the fault detection model inputs
    # ... (inside predict_fault_status function) ...

    # --- Extract Actual Outputs (Measurements) from current_net ---
    # This directly populates actual_outputs_from_pandapower_list in the correct order
    # with 0.0 for missing measurements (e.g., from out-of-service lines).
    actual_outputs_from_pandapower_list = []

    # Pre-fetch results for efficiency, and handle potential missing lines gracefully
    bus_results = current_net.res_bus
    line_results = current_net.res_line

    for col_name in original_nn_output_columns:
        value = 0.0  # Default value if measurement not found

        # Determine if it's a bus measurement or a line measurement
        if col_name.startswith('bus_'):
            parts = col_name.split('_')
            bus_id = int(parts[1])
            measure_type = '_'.join(parts[2:])  # e.g., 'vm_pu' or 'va_degree'
            if bus_id in bus_results.index:  # Bus always exists in results if power flow converged
                value = bus_results.at[bus_id, measure_type]
        elif col_name.startswith('line_'):
            parts = col_name.split('_')
            line_id = int(parts[1])
            measure_type = '_'.join(parts[2:])  # e.g., 'p_from_mw', 'q_from_mvar' etc.
            if line_id in line_results.index:  # Check if the line is in service and thus in results
                value = line_results.at[line_id, measure_type]

        actual_outputs_from_pandapower_list.append(value)

    # Create DataFrame for actual outputs from the carefully constructed list
    actual_outputs_from_pandapower_df = pd.DataFrame([actual_outputs_from_pandapower_list],
                                                     columns=original_nn_output_columns)

    # ... (rest of the predict_fault_status function, including your debug prints) ...
    print(
        f"--- DEBUG: Actual Measurements from Pandapower (first 5 bus/line output cols):\n{actual_outputs_from_pandapower_df.iloc[:, :5]}")

    # ... (rest of the predict_fault_status function) ...

    #calculate residuals (absolute difference) for input feature for classifier.
    scaled_actual_outputs = loaded_output_scaler.transform(actual_outputs_from_pandapower_df)
    residuals_np = np.abs(nn_predicted_outputs_scaled - scaled_actual_outputs)
    print('residuals_np:', residuals_np)

    # Create a DataFrame of residuals with appropriate column names
    residual_feature_names = [f'residual_{col}' for col in original_nn_output_columns]
    residuals_df = pd.DataFrame(residuals_np, columns=residual_feature_names)

    #creating X df for the detection model.
    X_for_inference = pd.DataFrame(index=[0])  # Create a single-row DataFrame

    # Add original inputs (from current_inputs_dict)
    for col in original_nn_input_columns:
        X_for_inference[f'input_orig_{col}'] = current_inputs_dict.get(col, 0.0)

    # Add actual outputs
    for col in original_nn_output_columns:
        X_for_inference[f'actual_output_{col}'] = actual_outputs_from_pandapower_df[col].iloc[0]

    # Add NN predicted outputs
    for col in original_nn_output_columns:
        X_for_inference[f'nn_pred_normal_{col}'] = nn_predicted_outputs_original_df[col].iloc[0]

    # Add residuals
    for col in residual_feature_names:
        X_for_inference[col] = residuals_df[col].iloc[0]

    feature_columns = [col for col in X_for_inference.columns if not col.endswith('_label')]
    X_for_inference =  X_for_inference[feature_columns]

    #fault detection using the X_for_inference df.
    detection_res = fault_detection_model.predict(X_for_inference)
    print(detection_res)
    print('is faulted?')

    if (detection_res>=0.5):
        is_faulted = 1
    else:
        is_faulted = 0

    result = {'is_faulted': is_faulted}
    print(result)

    if is_faulted == 1:
        locate_res = fault_locate_model.predict(X_for_inference)
        print(locate_res)
        predicted_localization_probs = {}
        fault_locate_label_ids = fault_locate_model.label_classes()
        if hasattr(fault_locate_model,"label_classes") and fault_locate_label_ids is not None:
            # If label_classes are integers (which they are for line IDs), convert them
            class_labels = [int(x) for x in fault_locate_label_ids]
            # localization_probabilities[0] should be the probabilities for the first (and only) sample
            for i, prob in enumerate(locate_res[0]):
                predicted_localization_probs[class_labels[i]] = prob
        else:
            print(
                "Warning: Could not retrieve label_classes from localization model. Probabilities might not be mapped correctly.")
            # Fallback if label_classes are not directly accessible this way
            predicted_localization_probs = {f'class_{i}': prob for i, prob in enumerate(locate_res[0])}

        # Find the most likely fault location
        most_likely_fault_id = max(predicted_localization_probs, key=predicted_localization_probs.get)
        most_likely_fault_prob = predicted_localization_probs[most_likely_fault_id]

        result["fault_localization"] = {
        "predicted_line_id": most_likely_fault_id,
        "probability": most_likely_fault_prob,
        "all_probabilities": predicted_localization_probs
        }
    else:
        result["fault_localization"] = "N/A (No fault detected)"
    return result

if __name__ == '__main__':
    # print("\nSimulating a healthy network state...")
    # healthy_net = create_cigre_network_mv()
    # # Run power flow for the healthy state to get measurements
    # pp.runpp(healthy_net)
    #
    # if healthy_net.converged:
    #     result_healthy = predict_fault_status(healthy_net)
    #     print(f"Inference Result (Healthy): {result_healthy}")
    # else:
    #     print("Healthy network scenario power flow did not converge. Skipping.")



    # --- Faulty network scenario---
    print("\nSimulating a faulted network state")
    faulted_net = create_cigre_network_mv() # start with a fresh net for this scenario
    faulted_line_id = 4 # simulating a fault on line 7
    faulted_net.line.at[faulted_line_id, 'in_service'] = False # deactivate the line
    pp.runpp(faulted_net) # run power flow for the faulted state to get measurements

    if faulted_net.converged:
        result_faulted = predict_fault_status(faulted_net)
        print('prediction:')
        print(result_faulted['fault_localization']['predicted_line_id'])
        print(f'probability: {result_faulted["fault_localization"]["probability"]}')
        print(f"Inference Result (Faulted on Line {faulted_line_id}): {result_faulted}")
    else:
        print(f"Faulted network scenario (Line {faulted_line_id}) power flow did not converge. Skipping.")

    print("\nSample inference tests complete.")