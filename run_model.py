import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import os
import pdb
import pandas as pd
import matplotlib.pyplot as plt

from nn_functions import load_model_pipeline, predict, calculate_solidity

if __name__ == '__main__':
    # --- Main execution block ---
    # This demonstrates how to use the functions to make a prediction.
    
    # 1. Load the trained model and scalers
    pipeline = load_model_pipeline(model_dir=".\model_weights")
    
    if pipeline[0] is not None:
        model, misc_scaler, y_scaler, meta = pipeline

        # 2. Define example data for a single propeller
        # IMPORTANT: The lengths of the arrays MUST match the model's training data.
        # Check the N_radial and N_adv values in 'meta.json'.
        N_radial_expected = meta['N_radial']
        N_adv_expected = meta['N_adv']

        print(f"\nThis model expects geometry arrays of length {N_radial_expected} and performance arrays of length {N_adv_expected}.")

        # Example geometry and operating conditions
        # apc29ff_9x5_geom
        example_radius = np.linspace(0.15, 1.0, N_radial_expected)
        
        norm_chord = 0.5
        norm_twist = 100
        unsampled_chord = np.array([0.160, 0.146, 0.144, 0.143, 0.143, 0.146, 0.151, 0.155, 0.158, 0.160, 0.159, 0.155, 0.146, 0.133, 0.114, 0.089, 0.056, 0.022]) / norm_chord
        example_chord = np.interp(np.linspace(0, 1, N_radial_expected), np.linspace(0, 1, len(unsampled_chord)), unsampled_chord)
        
        unsampled_twist = np.array([31.68, 34.45, 35.93, 33.33, 29.42, 26.25, 23.67, 21.65, 20.02, 18.49, 17.06, 15.95, 14.87, 13.82, 12.77, 11.47, 10.15, 8.82]) / norm_twist

        example_twist = np.interp(np.linspace(0, 1, N_radial_expected), np.linspace(0, 1, len(unsampled_twist)), unsampled_twist)
        example_diameter = 5 * 0.0254
        example_n_blades = 2
        example_j_array = np.linspace(0.1, 0.8, N_adv_expected)
        example_tip_mach = 0.21

        example_solidity = calculate_solidity(example_chord, example_diameter, example_n_blades)


        # 3. Run the prediction
        predicted_ct, predicted_cp = predict(
            model, misc_scaler, y_scaler,
            example_radius,
            example_chord,
            example_twist,
            example_n_blades,
            example_diameter,
            example_j_array
        )

        # 4. Print the results
        print("\n--- Prediction Results ---")
        print(f"For a propeller with Diameter={example_diameter}:")
        
        results_df = pd.DataFrame({
            'J (Advance Ratio)': example_j_array,
            'Predicted CT': predicted_ct,
            'Predicted CP': predicted_cp
        })
        print(results_df.to_string())
        
        # Example of how to add pandas if not already there
        try:
            import pandas as pd
        except ImportError:
            print("\nNote: For a nicer output format, please install pandas (`pip install pandas`).")
            # Fallback to a simpler print if pandas is not available
            for j, ct, cp in zip(example_j_array, predicted_ct, predicted_cp):
                print(f"J={j:.3f} -> Predicted CT={ct:.4f}, Predicted CP={cp:.4f}")

        # Create a DataFrame for the actual data
        actual_data = pd.DataFrame({
            'J': [0.401, 0.425, 0.444, 0.474, 0.493, 0.519, 0.543, 0.569, 0.589, 0.612, 0.641],
            'CT': [0.0396, 0.0352, 0.0317, 0.0266, 0.0232, 0.0181, 0.0138, 0.0088, 0.0050, 0.0006, -0.0054],
            'CP': [0.0279, 0.0263, 0.0250, 0.0230, 0.0217, 0.0196, 0.0177, 0.0154, 0.0136, 0.0114, 0.0083],
            'eta': [0.568, 0.568, 0.563, 0.548, 0.527, 0.478, 0.424, 0.326, 0.217, 0.033, -0.418]
        })
 

        plt.figure(figsize=(8, 5))
        plt.plot(actual_data['J'], actual_data['CT'], label='APC Wind tunnel data', marker='o')
        plt.plot(results_df['J (Advance Ratio)'], results_df['Predicted CT'], label='APC_NN', marker='x')
        plt.xlabel('Advance Ratio (J)')
        plt.ylabel('Thrust Coefficient (CT)')
        plt.title('APC Wind tunnel data vs APC_NN')
        plt.legend()
        # No grid lines
        plt.show()

        pdb.set_trace()
