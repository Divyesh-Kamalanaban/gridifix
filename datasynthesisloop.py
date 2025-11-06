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

def datasynthesiser():
    # try excerpt loading the saved NN model (h5) and the scalers (pkl)
    try:
        loaded_model = tf.keras.models.load_model('power_flow_nn_model.h5')
        loaded_input_scaler = joblib.load('input_scaler.pkl')
        loaded_output_scaler = joblib.load('output_scaler.pkl')
        print("Successfully loaded power flow NN model and scalers in main script.")

        # --- NEW DEBUG PRINTS FOR SCALER FEATURE NAMES ---
        if hasattr(loaded_input_scaler, 'feature_names_in_'):
            print(f"DEBUG: loaded_input_scaler expects {len(loaded_input_scaler.feature_names_in_)} input features.")
            print(f"DEBUG: loaded_input_scaler feature names (first 5): {loaded_input_scaler.feature_names_in_[:5]}")
            print(f"DEBUG: loaded_input_scaler feature names (last 5): {loaded_input_scaler.feature_names_in_[-5:]}")
        else:
            print(
                "DEBUG: loaded_input_scaler does not have 'feature_names_in_' attribute (might be older scikit-learn version or custom scaler).")

        if hasattr(loaded_output_scaler, 'feature_names_in_'):
            print(f"DEBUG: loaded_output_scaler expects {len(loaded_output_scaler.feature_names_in_)} output features.")
            print(f"DEBUG: loaded_output_scaler feature names (first 5): {loaded_output_scaler.feature_names_in_[:5]}")
            print(f"DEBUG: loaded_output_scaler feature names (last 5): {loaded_output_scaler.feature_names_in_[-5:]}")
        else:
            print(
                "DEBUG: loaded_output_scaler does not have 'feature_names_in_' attribute (might be older scikit-learn version or custom scaler).")
        # --- END NEW DEBUG PRINTS ---

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Required model or scaler file not found: {e}")
        print(
            "Please ensure 'power_flow_nn_model.h5', 'input_scaler.pkl', and 'output_scaler.pkl' are in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred while loading NN model or scalers: {e}")
        sys.exit(1)

    # load the csvs and grab the column names from them - to avoid feature mismatches.
    try:
        temp_inputs_df_for_cols = pd.read_csv("combined_power_flow_inputs.csv")
        temp_outputs_df_for_cols = pd.read_csv("combined_power_flow_outputs.csv")

        print("\n--- Identifying and Removing Constant Columns for Column Name Extraction ---")

        constant_input_cols_for_nn = [col for col in temp_inputs_df_for_cols.columns if
                                      temp_inputs_df_for_cols[col].nunique() == 1]
        if constant_input_cols_for_nn:
            print(
                f"Found constant input columns in original data: {constant_input_cols_for_nn}. Excluding them from NN input features.")
            original_nn_input_columns = temp_inputs_df_for_cols.drop(
                columns=constant_input_cols_for_nn).columns.tolist()
        else:
            print("No constant input columns found in original data. Using all input columns.")
            original_nn_input_columns = temp_inputs_df_for_cols.columns.tolist()

        constant_output_cols_for_nn = [col for col in temp_outputs_df_for_cols.columns if
                                       temp_outputs_df_for_cols[col].nunique() == 1]
        if constant_output_cols_for_nn:
            print(
                f"Found constant output columns in original data: {constant_output_cols_for_nn}. Excluding them from NN output features.")
            original_nn_output_columns = temp_outputs_df_for_cols.drop(
                columns=constant_output_cols_for_nn).columns.tolist()
        else:
            print("No constant output columns found in original data. Using all output columns.")
            original_nn_output_columns = temp_outputs_df_for_cols.columns.tolist()

        print(f"Final count of NN input columns (after preprocessing): {len(original_nn_input_columns)}")
        print(f"Final count of NN output columns (after preprocessing): {len(original_nn_output_columns)}")
        print(f"Sample NN input columns (first 5): {original_nn_input_columns[:5]}")
        print(f"Sample NN input columns (last 5): {original_nn_input_columns[-5:]}")
        print(f"Sample NN output columns (first 5): {original_nn_output_columns[:5]}")
        print(f"Sample NN output columns (last 5): {original_nn_output_columns[-5:]}")

        print("Successfully determined NN training column names after preprocessing.")

    except FileNotFoundError as e:
        print(f"FATAL ERROR: Original NN training data CSVs not found: {e}")
        print("Please ensure 'combined_power_flow_inputs.csv' and 'combined_power_flow_outputs.csv' are available.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: An error occurred while processing original NN training column names: {e}")
        sys.exit(1)

    # --- Your existing power system analysis steps (unchanged) ---
    net = initcigre.init_cigre()
    A, B, C, D, pv_bus, pq_bus, sl_bus, num_total_buses = dlpfcoeff.get_dlpf_coeff(net)
    dlpf_vm_pu, dlpf_theta_rad = solvedlpf.solve_dlpf(net, A, B, C, D, pq_bus, pv_bus, sl_bus, num_total_buses)
    erroranalysisdlpf.error_analysis_dlpf(net, dlpf_vm_pu, dlpf_theta_rad)
    dermodelling.der_modelling(net, {'type': 'gen', 'bus': 5, 'p_mw': 0.2, 'vm_pu': 1.005}, A, B, C, D, pq_bus, pv_bus,
                               sl_bus, num_total_buses)

    # --- Data Generation Parameters ---
    num_samples_per_sgen_config = 500
    load_range = (0.7, 1.3)
    sgen_p_range = (0.0, 0.8)
    sgen_q_factor_range = (-0.5, 0.5)
    fault_prob = 0.15

    sgen_configs_to_iterate = [
        {'bus': 1, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 2, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 3, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 4, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 5, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 6, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 7, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 8, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 9, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 10, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 11, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 12, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 13, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
        {'bus': 14, 'p_mw_range': sgen_p_range, 'q_mvar_range_factor': sgen_q_factor_range},
    ]

    all_generated_data_frames = []

    print("\nStarting comprehensive data generation across sgen locations...")

    # iterates over the configs list to create samples of each config; along with the fault probabilities too.
    for s_config in sgen_configs_to_iterate:
        data_for_current_sgen_config = samplegenerator.generate_samples(
            nn_model=loaded_model,
            input_scaler=loaded_input_scaler,
            output_scaler=loaded_output_scaler,
            nn_input_columns=original_nn_input_columns,
            nn_output_columns=original_nn_output_columns,
            num_samples=num_samples_per_sgen_config,
            load_variation_range=load_range,
            sgen_config=s_config,
            fault_probability=fault_prob
        )
        all_generated_data_frames.append(data_for_current_sgen_config)

    # concating the dfs and converting it into a df.
    final_detection_dataset_df = pd.concat(all_generated_data_frames, ignore_index=True)

    print("\n\n####################################################################################################")
    print("### Final Combined Fault Detection Dataset Summary ###")
    print("####################################################################################################")
    print(f"Total Samples Generated: {len(final_detection_dataset_df)}")
    print(f"Final DataFrame shape: {final_detection_dataset_df.shape}")

    print("\n--- Sample of Final Fault Detection Dataset (first 2 rows) ---")
    print(final_detection_dataset_df.head(2))

    print("\n--- Value counts for 'is_faulted_label' ---")
    print(final_detection_dataset_df['is_faulted_label'].value_counts())

    print("\n--- Value counts for 'fault_location_id_label' (top 5) ---")
    print(final_detection_dataset_df['fault_location_id_label'].value_counts().head())

    try:
        final_detection_dataset_df.to_csv("fault_detection_dataset_full.csv", index=False)
        print("\nFinal fault detection dataset successfully saved to fault_detection_dataset.csv")
    except Exception as e:
        print(f"\nError saving final fault detection dataset to CSV: {e}")

    print("\nAll data generation for fault detection/localization complete.")