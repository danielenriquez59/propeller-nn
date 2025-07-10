import torch
import torch.nn as nn
import numpy as np
import joblib
import json
import os

class PropNet(nn.Module):
    """
    Neural network for propeller performance prediction.
    This class definition must match the one used during training.
    """
    def __init__(self, n_radial, n_adv, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)            # → [B, 64, 1]
        )
        in_dense = 64 + 3 + n_adv              # 64 CNN feat + scalars + J array
        self.mlp = nn.Sequential(
            nn.Linear(in_dense, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 2*n_adv)
        )

    def forward(self, x_geom, x_misc):
        # x_geom : [B, 3, N_radial]   (radius, chord, twist stacked on channel dim)
        z = self.cnn(x_geom).squeeze(-1)       # [B, 64]
        z = torch.cat([z, x_misc], dim=1)      # misc = [diam, N_blades, tip_mach, J...]
        return self.mlp(z)

def load_model_pipeline(model_dir="."):
    """
    Loads the complete model pipeline from saved files.
    
    Args:
        model_dir (str): Directory where model files are stored.

    Returns:
        A tuple containing:
        - model: The loaded PyTorch model.
        - misc_scaler: The fitted miscellaneous feature scaler.
        - y_scaler: The fitted target scaler.
        - meta: A dictionary of metadata (N_radial, N_adv, etc.).
    """
    print(f"Loading model components from directory: {model_dir}")
    try:
        # Load metadata
        with open(os.path.join(model_dir, "meta.json"), "r") as f:
            meta = json.load(f)
        
        # Load scalers
        misc_scaler = joblib.load(os.path.join(model_dir, "misc_scaler.joblib"))
        y_scaler = joblib.load(os.path.join(model_dir, "y_scaler.joblib"))
        
        # Determine model dimensions from metadata
        N_radial = meta['N_radial']
        N_adv = meta['N_adv']

        # Create model instance and load weights
        model = PropNet(n_radial=N_radial, n_adv=N_adv)
        model.load_state_dict(torch.load(os.path.join(model_dir, "propnet_weights.pt")))
        model.eval()  # Set model to evaluation mode

        print("Model pipeline loaded successfully.")
        return model, misc_scaler, y_scaler, meta

    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure 'meta.json', 'misc_scaler.joblib', 'y_scaler.joblib', and 'propnet_weights.pt' are in the specified directory.")
        return None, None, None, None

def build_input_tensors(radius, chord, twist, diameter, N_blades, J_array, tip_mach):
    """Build the geometry and miscellaneous tensors for a single prediction."""
    # Add a batch dimension of 1
    x_geom = np.zeros((1, 3, len(radius)), dtype=np.float32)
    x_geom[0, 0, :] = np.asarray(radius, dtype=np.float32)
    x_geom[0, 1, :] = np.asarray(chord, dtype=np.float32)
    x_geom[0, 2, :] = np.asarray(twist, dtype=np.float32)

    misc_scalars = np.array([diameter, N_blades, tip_mach], dtype=np.float32)
    j_array_np = np.asarray(J_array, dtype=np.float32)
    x_misc = np.concatenate([misc_scalars, j_array_np]).reshape(1, -1)
    
    return x_geom, x_misc

def predict(model, misc_scaler, y_scaler, radius, chord, twist, diameter, N_blades, J_array, tip_mach):
    """
    Makes a performance prediction for a single propeller.
    
    Args:
        model, misc_scaler, y_scaler: Loaded pipeline components.
        All other args: Propeller geometry and operating conditions.

    Returns:
        A tuple of (ct_pred, cp_pred) numpy arrays.
    """
    # 1. Build the input tensors
    x_geom, x_misc = build_input_tensors(radius, chord, twist, diameter, N_blades, J_array, tip_mach)
    
    # 2. Scale the miscellaneous features
    x_misc_scaled = misc_scaler.transform(x_misc)
    
    # 3. Run prediction
    with torch.no_grad():
        y_hat_scaled = model(torch.tensor(x_geom), torch.tensor(x_misc_scaled)).numpy()
    
    # 4. Inverse transform the prediction to get actual values
    y_hat = y_scaler.inverse_transform(y_hat_scaled)
    
    # 5. Reshape back into separate CT and CP arrays
    N_adv = len(J_array)
    ct_pred, cp_pred = y_hat.reshape(2, N_adv)
    
    return ct_pred, cp_pred
