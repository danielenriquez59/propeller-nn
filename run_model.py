import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import os
import pdb
import pandas as pd
import matplotlib.pyplot as plt

class PropNet(nn.Module):
    """
    Neural network for propeller performance prediction.
    This class definition must match the one used during training.
    """
    def __init__(self, in_dim, hidden=(256, 128, 64), out_dim=2, p_drop=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        self.net = nn.Sequential(*layers, nn.Linear(prev, out_dim))

    def forward(self, x):
        return self.net(x)

def load_model_pipeline(model_dir="."):
    """
    Loads the complete model pipeline from saved files.
    
    Args:
        model_dir (str): Directory where model files are stored.

    Returns:
        A tuple containing:
        - model: The loaded PyTorch model.
        - x_scaler: The fitted feature scaler.
        - y_scaler: The fitted target scaler.
        - meta: A dictionary of metadata (N_radial, N_adv, etc.).
    """
    print(f"Loading model components from directory: {model_dir}")
    try:
        # Load metadata
        with open(os.path.join(model_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        
        # Load scalers
        x_scaler = joblib.load(os.path.join(model_dir, "x_scaler.joblib"))
        y_scaler = joblib.load(os.path.join(model_dir, "y_scaler.joblib"))
        
        # Determine model dimensions from metadata
        N_radial = meta['N_radial']
        N_adv = meta['N_adv']
        input_dim = 3 * N_radial + 2 + N_adv
        output_dim = 2 * N_adv

        # Create model instance and load weights
        model = PropNet(in_dim=input_dim, out_dim=output_dim)
        model.load_state_dict(torch.load(os.path.join(model_dir, "propnet_weights.pt")))
        model.eval()  # Set model to evaluation mode

        print("Model pipeline loaded successfully.")
        return model, x_scaler, y_scaler, meta

    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure 'meta.json', 'x_scaler.joblib', 'y_scaler.joblib', and 'propnet_weights.pt' are in the specified directory.")
        return None, None, None, None

def build_feature_vector(radius, chord, twist, diameter, N_blades, J_array):
    """
    Builds the flattened feature vector for a single sample.
    This must match the vector structure used during training.
    """
    geom = np.concatenate([
        np.asarray(radius, dtype=np.float32),
        np.asarray(chord, dtype=np.float32),
        np.asarray(twist, dtype=np.float32)
    ])
    scalars = np.array([diameter, N_blades], dtype=np.float32)
    j_array_np = np.asarray(J_array, dtype=np.float32)
    
    features = np.concatenate([geom, scalars, j_array_np])
    return features.astype(np.float32)

def predict(model, x_scaler, y_scaler, radius, chord, twist, diameter, N_blades, J_array):
    """
    Makes a performance prediction for a single propeller.
    
    Args:
        model, x_scaler, y_scaler: Loaded pipeline components.
        All other args: Propeller geometry and operating conditions.

    Returns:
        A tuple of (ct_pred, cp_pred) numpy arrays.
    """
    # 1. Build the feature vector
    x = build_feature_vector(radius, chord, twist, diameter, N_blades, J_array)
    
    # 2. Scale the feature vector (it expects a 2D array, so we add a batch dimension)
    x_scaled = x_scaler.transform(x[None, :])
    
    # 3. Run prediction
    with torch.no_grad():
        y_hat_scaled = model(torch.tensor(x_scaled)).numpy()
    
    # 4. Inverse transform the prediction to get actual values
    y_hat = y_scaler.inverse_transform(y_hat_scaled)
    
    # 5. Reshape back into separate CT and CP arrays
    N_adv = len(J_array)
    ct_pred, cp_pred = y_hat.reshape(2, N_adv)
    
    return ct_pred, cp_pred

if __name__ == '__main__':
    # --- Main execution block ---
    # This demonstrates how to use the functions to make a prediction.
    
    # 1. Load the trained model and scalers
    pipeline = load_model_pipeline()
    
    if pipeline[0] is not None:
        model, x_scaler, y_scaler, meta = pipeline

        # 2. Define example data for a single propeller
        # IMPORTANT: The lengths of the arrays MUST match the model's training data.
        # Check the N_radial and N_adv values in 'meta.json'.
        N_radial_expected = meta['N_radial']
        N_adv_expected = meta['N_adv']

        print(f"\nThis model expects geometry arrays of length {N_radial_expected} and performance arrays of length {N_adv_expected}.")

        # Example geometry and operating conditions
        example_radius = np.linspace(0.15, 1.0, N_radial_expected)

        unsampled_chord = np.array([0.160, 0.146, 0.144, 0.143, 0.143, 0.146, 0.151, 0.155, 0.158, 0.160, 0.159, 0.155, 0.146, 0.133, 0.114, 0.089, 0.056, 0.022])
        example_chord = np.interp(np.linspace(0, 1, N_radial_expected), np.linspace(0, 1, len(unsampled_chord)), unsampled_chord)
        
        unsampled_twist = np.array([31.68, 34.45, 35.93, 33.33, 29.42, 26.25, 23.67, 21.65, 20.02, 18.49, 17.06, 15.95, 14.87, 13.82, 12.77, 11.47, 10.15, 8.82])

        example_twist = np.interp(np.linspace(0, 1, N_radial_expected), np.linspace(0, 1, len(unsampled_twist)), unsampled_twist)
        example_diameter = 5
        example_n_blades = 2
        example_j_array = np.linspace(0.1, 0.8, N_adv_expected)

        # 3. Run the prediction
        predicted_ct, predicted_cp = predict(
            model, x_scaler, y_scaler,
            example_radius,
            example_chord,
            example_twist,
            example_diameter,
            example_n_blades,
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