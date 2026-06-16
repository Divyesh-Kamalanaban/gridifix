# samplegenerator.py (Revised - Explicitly filter output features)

import numpy as np
import pandapower as pp
import pandas as pd
from pandapower.networks import create_cigre_network_mv
import copy
import tensorflow as tf
import joblib


def generate_samples(
        nn_model,  # Required argument (no default)
        input_scaler,  # Required argument (no default)
        output_scaler,  # Required argument (no default)
        nn_input_columns,  # REQUIRED: Exact input column names from NN training (passed from main.py)
        nn_output_columns,  # REQUIRED: Exact output column names from NN training (passed from main.py)
        num_samples=1000,
        load_variation_range=(0.7, 1.3),
        sgen_config=None,
        fault_probability=0.1
):
    """
    Generates data samples for fault detection and localization.
    Ensures consistency with the feature names and counts used during NN training.
    """
    print(
        f"\n--- Generating {num_samples} Data Samples (with Faults) for sgen_config: {sgen_config['bus'] if sgen_config else 'None'} ---")

    base_net = create_cigre_network_mv()
    pp.runpp(base_net)

    orig_loads = base_net.load.copy()
    all_lines = base_net.line.index.to_list()

    combined_dataset_records = []

    bus_ids = base_net.bus.index.to_numpy()
    line_ids = base_net.line.index.to_numpy()

    # Define the *full* set of possible output columns as they come directly from pandapower.
    # This list will always have 90 features for CIGRE MV (15 buses * 2 + 15 lines * 4).
    # We need this to create an intermediate DataFrame with proper column names.
    full_pandapower_output_columns = []
    for bus_id in bus_ids:
        full_pandapower_output_columns.extend([f'bus_{bus_id}_vm_pu', f'bus_{bus_id}_va_degree'])
    for line_id in line_ids:
        full_pandapower_output_columns.extend([
            f'line_{line_id}_p_from_mw', f'line_{line_id}_q_from_mvar',
            f'line_{line_id}_p_to_mw', f'line_{line_id}_q_to_mvar'
        ])


    for i in range(num_samples):
        temp_net = copy.deepcopy(base_net)
        current_inputs_dict = {}

        # --- 1. Vary Loads and Sgens (Inputs for NN Prediction) ---
        for idx, load_rows in orig_loads.iterrows():
            p_scale = np.random.uniform(load_variation_range[0], load_variation_range[1])
            q_scale = np.random.uniform(load_variation_range[0], load_variation_range[1])

            temp_net.load.at[idx, 'p_mw'] = load_rows['p_mw'] * p_scale
            temp_net.load.at[idx, 'q_mvar'] = load_rows['q_mvar'] * q_scale

            current_inputs_dict[f'load_{idx}_p_mw'] = temp_net.load.at[idx, 'p_mw']
            current_inputs_dict[f'load_{idx}_q_mvar'] = temp_net.load.at[idx, 'q_mvar']

        if sgen_config:
            sgen_bus = sgen_config['bus']
            existing_sgens = temp_net.sgen[temp_net.sgen.bus == sgen_bus]
            for sgen_idx in existing_sgens.index:
                pp.remove_sgen(temp_net, sgen_idx)

            p_sgen = np.random.uniform(sgen_config['p_mw_range'][0], sgen_config['p_mw_range'][1])
            q_sgen_factor = np.random.uniform(sgen_config['q_mvar_range_factor'][0],
                                              sgen_config['q_mvar_range_factor'][1])
            q_sgen = p_sgen * q_sgen_factor

            pp.create_sgen(temp_net, bus=sgen_bus, p_mw=p_sgen, q_mvar=q_sgen)

            current_inputs_dict[f'sgen_bus_{sgen_bus}_p_mw'] = p_sgen
            current_inputs_dict[f'sgen_bus_{sgen_bus}_q_mvar'] = q_sgen

        # Prepare NN inputs (uses nn_input_columns passed from main.py)
        nn_input_df_row = pd.DataFrame([current_inputs_dict], columns=nn_input_columns).fillna(0)

        # --- 2. Predict Normal State with Your Existing Power Flow NN ---
        scaled_nn_inputs = input_scaler.transform(nn_input_df_row)
        nn_predicted_outputs_scaled = nn_model.predict(scaled_nn_inputs, verbose=0)
        nn_predicted_outputs_original = output_scaler.inverse_transform(nn_predicted_outputs_scaled)

        # fault logic - by default a line is not faulted and the loc-id is -1.
        is_faulted = 0
        fault_location_id = -1

        if np.random.rand() < fault_probability:
            is_faulted = 1
            if all_lines:
                faulted_line_id = np.random.choice(all_lines)
                temp_net.line.at[faulted_line_id, 'in_service'] = False
                fault_location_id = faulted_line_id
            else:
                is_faulted = 0
                fault_location_id = -1

        try:
            #pp analysis after sgen and line fault assembly.
            pp.runpp(temp_net)
            if not temp_net.converged:
                continue

            # runs only if converge =  true.

            # fill in the values from pp analysis
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

            # Convert the full 90-feature list into a DataFrame with its expected column names
            full_actual_outputs_df = pd.DataFrame([full_actual_outputs_list], columns=full_pandapower_output_columns)

            # selects only the nn_output_cols in the full_actual_outputs_df.
            actual_outputs_for_scaler = full_actual_outputs_df[nn_output_columns].to_numpy()


            # calculate residuals or the absolute diff between the outputs
            scaled_actual_outputs = output_scaler.transform(actual_outputs_for_scaler)  # Use the filtered array here
            residuals = np.abs(nn_predicted_outputs_scaled - scaled_actual_outputs)

            # this record is for each sample of the network if that makes sense.
            record = {}

            # Add original inputs (features for classifier) - iterate over `nn_input_columns`
            for col_name in nn_input_columns:
                record[f'input_orig_{col_name}'] = current_inputs_dict.get(col_name, 0.0)

            # Add actual outputs (simulated measurements) - iterate over `nn_output_columns`
            for col_idx, col_name in enumerate(nn_output_columns):
                # Ensure we're taking from the correctly filtered `actual_outputs_for_scaler`
                record[f'actual_output_{col_name}'] = actual_outputs_for_scaler[0, col_idx]

            # Add NN predicted outputs (the 'normal' baseline) - iterate over `nn_output_columns`
            for col_idx, col_name in enumerate(nn_output_columns):
                record[f'nn_pred_normal_{col_name}'] = nn_predicted_outputs_original[0, col_idx]

            # Add residuals (CRITICAL FEATURES FOR CLASSIFIERS) - iterate over `nn_output_columns`
            for col_idx, col_name in enumerate(nn_output_columns):
                record[f'residual_{col_name}'] = residuals[0, col_idx]

            # Add true labels for this sample
            record['is_faulted_label'] = is_faulted
            record['fault_location_id_label'] = fault_location_id

            combined_dataset_records.append(record)

        except Exception as e:
            print(f"Error during pandapower run or data collection for sample {i + 1}: {e}. Skipping.")
            continue

        if (i + 1) % 500 == 0:
            #acts as a progress bar.
            print(f"  Generated {len(combined_dataset_records)} valid samples so far for current sgen config.")

    return pd.DataFrame(combined_dataset_records)