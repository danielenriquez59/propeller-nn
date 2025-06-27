import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import ast, os
import pdb

out_dir = "model_weights"

def diagnose_data_homogeneity(df):
    """
    Checks for consistent array lengths across all rows in the DataFrame.

    This function iterates through the DataFrame and identifies any rows
    where the lengths of the geometry or performance arrays differ from
    the lengths in the first row. It then prints a detailed report of
    any inconsistencies found.

    Returns
    -------
    bool
        True if the data is homogeneous, False otherwise.
    """
    if df.empty:
        print("Diagnostic check: DataFrame is empty. No data to check.")
        return True

    print("\n--- Running Data Homogeneity Diagnostics ---")
    
    # Use the first row as the reference for expected lengths
    first_row = df.iloc[0]
    expected_lengths = {
        'radius': len(first_row['radius']),
        'chord': len(first_row['chord']),
        'twist': len(first_row['twist']),
        'J': len(first_row['J']),
        'CT': len(first_row['CT']),
        'CP': len(first_row['CP']),
    }
    print(f"Expected array lengths based on first row (prop_id {first_row['propeller_names']}):")
    for key, length in expected_lengths.items():
        print(f"  - {key}: {length}")

    # Store inconsistencies found
    inconsistencies = []

    # Iterate over all rows and check lengths
    for index, row in df.iterrows():
        current_lengths = {
            'radius': len(row['radius']),
            'chord': len(row['chord']),
            'twist': len(row['twist']),
            'J': len(row['J']),
            'CT': len(row['CT']),
            'CP': len(row['CP']),
        }

        # Compare current lengths with expected lengths
        for key, expected_len in expected_lengths.items():
            actual_len = current_lengths[key]
            if actual_len != expected_len:
                inconsistencies.append({
                    'row_index': index,
                    'prop_id': row['prop_id'],
                    'column': key,
                    'expected': expected_len,
                    'actual': actual_len,
                })

    # Print the report
    if not inconsistencies:
        print("\n[SUCCESS] All array lengths are consistent across the dataset.")
        print("------------------------------------------\n")
        return True
    else:
        print(f"\n[FAILURE] Found {len(inconsistencies)} inconsistencies in array lengths.")
        # Create a DataFrame for easier viewing
        report_df = pd.DataFrame(inconsistencies)
        print("Summary of inconsistencies:")
        print(report_df.to_string())
        print("------------------------------------------\n")
        return False
    
def read_data():
    """
    Reads and merges propeller data without flattening performance arrays.

    For each row in the `performance` DataFrame, this function looks up
    and appends the corresponding metadata and full geometry arrays.

    The final DataFrame will have one row for each original performance
    record, with columns containing arrays for J, CT, CP, radius, chord,
    and twist.
    """
    
    df = pd.read_pickle("propeller_data_combined.pkl")

    # These definitions will need to be handled by a different model architecture
    feature_cols = ["radius", "chord", "twist", "diameter", "N_blades", "J"]
    target_cols = ["CT", "CP"]
    N_radial = len(df.iloc[0]["radius"])
    
    return df, feature_cols, target_cols, N_radial


def prepare_data(df, feature_cols, target_cols):
    """
    Transforms the DataFrame into structured numpy arrays for CNN-based training.

    This function creates three sets of arrays:
    1. X_geom: A 3D array for the CNN (radius, chord, twist).
       Shape: (n_samples, 3, N_radial)
    2. X_misc: A 2D array for the MLP (diameter, N_blades, J_array).
       Shape: (n_samples, 2 + N_adv)
    3. y: A 2D array for the targets (CT, CP).
       Shape: (n_samples, 2 * N_adv)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot prepare data.")
    
    N_radial = len(df.iloc[0]["radius"])
    N_adv = len(df.iloc[0]["J"])

    # Pre-allocate numpy arrays for efficiency
    n_samples = len(df)
    X_geom = np.zeros((n_samples, 3, N_radial), dtype=np.float32)
    X_misc = np.zeros((n_samples, 2 + N_adv), dtype=np.float32)
    y = np.zeros((n_samples, 2 * N_adv), dtype=np.float32)

    # Populate the arrays
    for i, (_, row) in enumerate(df.iterrows()):
        X_geom[i, 0, :] = np.asarray(row["radius"], dtype=np.float32)
        X_geom[i, 1, :] = np.asarray(row["chord"], dtype=np.float32)
        X_geom[i, 2, :] = np.asarray(row["twist"], dtype=np.float32)
        
        misc_scalars = np.array([row["diameter"], row["N_blades"]], dtype=np.float32)
        j_array = np.asarray(row["J"], dtype=np.float32)
        X_misc[i, :] = np.concatenate([misc_scalars, j_array])

        y[i, :] = np.concatenate([
            np.asarray(row["CT"], dtype=np.float32),
            np.asarray(row["CP"], dtype=np.float32)
        ])
    
    # Split: 70% train, 15% val, 15% test
    (X_geom_train, X_geom_tmp,
     X_misc_train, X_misc_tmp,
     y_train, y_tmp) = train_test_split(X_geom, X_misc, y, test_size=0.3, random_state=0)
    
    (X_geom_val, X_geom_test,
     X_misc_val, X_misc_test,
     y_val, y_test) = train_test_split(X_geom_tmp, X_misc_tmp, y_tmp, test_size=0.5, random_state=0)

    # Fit and apply scaler only on misc features
    misc_scaler = StandardScaler().fit(X_misc_train)
    X_misc_train = misc_scaler.transform(X_misc_train)
    X_misc_val = misc_scaler.transform(X_misc_val)
    X_misc_test = misc_scaler.transform(X_misc_test)

    # Fit and apply scaler on targets
    y_scaler = StandardScaler().fit(y_train)
    y_train = y_scaler.transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    
    return (X_geom_train, X_misc_train, X_geom_val, X_misc_val, X_geom_test, X_misc_test,
            y_train, y_val, y_test, misc_scaler, y_scaler)


def create_data_loaders(X_geom, X_misc, y, batch_size=256):
    """Create PyTorch data loaders for training and validation."""
    dataset = TensorDataset(
        torch.tensor(X_geom),
        torch.tensor(X_misc),
        torch.tensor(y)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class PropNet(nn.Module):
    def __init__(self, n_radial, n_adv, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)            # → [B, 64, 1]
        )
        in_dense = 64 + 2 + n_adv              # 64 CNN feat + scalars + J array
        self.mlp = nn.Sequential(
            nn.Linear(in_dense, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 2*n_adv)
        )

    def forward(self, x_geom, x_misc):
        # x_geom : [B, 3, N_radial]   (radius, chord, twist stacked on channel dim)
        z = self.cnn(x_geom).squeeze(-1)       # [B, 64]
        z = torch.cat([z, x_misc], dim=1)      # misc = [diam, N_blades, J...]
        return self.mlp(z)


def create_model(n_radial, n_adv):
    """Create and initialize the PropNet model."""
    model = PropNet(n_radial=n_radial, n_adv=n_adv)
    return model


def train_model(model,train_loader,X_geom_val,X_misc_val,y_val,epochs=1000,lr=2e-3,weight_decay=1e-4,patience=300):
    """
    Train the model with early stopping and an adaptive learning rate.

    This function uses a ReduceLROnPlateau scheduler to decrease the learning
    rate when the validation loss plateaus.

    Parameters:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): PyTorch DataLoader yielding training batches.
        X_geom_val (np.ndarray): Validation geometry input, shape [N, 3, N_radial].
        X_misc_val (np.ndarray): Validation misc input, shape [N, 2+N_adv].
        y_val (np.ndarray): Validation targets, shape [N, 2*N_adv].
        epochs (int): Maximum number of training epochs.
        lr (float): Initial learning rate for Adam optimizer.
        weight_decay (float): L2 regularization strength.
        patience (int): Early stopping patience (epochs to wait for improvement).
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.2, patience=50)
    
    # Move validation data to tensor once
    X_geom_val_t = torch.tensor(X_geom_val)
    X_misc_val_t = torch.tensor(X_misc_val)
    y_val_t = torch.tensor(y_val)

    best_val, wait = np.inf, 0
    
    for epoch in range(epochs):
        # ---- train one epoch ----
        model.train()
        for x_geom_b, x_misc_b, yb in train_loader:
            pred = model(x_geom_b, x_misc_b)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            val_pred = model(X_geom_val_t, X_misc_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        
        print(f"epoch {epoch:3d} | val MSE {val_loss:.4g}")

        # Step the scheduler
        scheduler.step(val_loss)

        if val_loss < best_val:
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
    
    return model


def evaluate_model(model, X_geom_test, X_misc_test, y_test, y_scaler, N_adv):
    """Evaluate the trained model on test set."""
    # Load best model weights
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
    model.eval()
    
    with torch.no_grad():
        pred_scaled = model(torch.tensor(X_geom_test), torch.tensor(X_misc_test)).numpy()
    
    # Invert scaling to get actual values
    y_pred = y_scaler.inverse_transform(pred_scaled)
    y_true = y_scaler.inverse_transform(y_test)

    # Reshape to separate CT and CP
    y_pred_reshaped = y_pred.reshape(-1, 2, N_adv)
    y_true_reshaped = y_true.reshape(-1, 2, N_adv)

    # Calculate metrics for CT and CP separately
    mae_ct = mean_absolute_error(y_true_reshaped[:, 0, :], y_pred_reshaped[:, 0, :])
    mae_cp = mean_absolute_error(y_true_reshaped[:, 1, :], y_pred_reshaped[:, 1, :])
    r2_ct = r2_score(y_true_reshaped[:, 0, :], y_pred_reshaped[:, 0, :])
    r2_cp = r2_score(y_true_reshaped[:, 1, :], y_pred_reshaped[:, 1, :])

    print("MAE CT      :", mae_ct)
    print("MAE CP      :", mae_cp)
    print("R² CT       :", r2_ct)
    print("R² CP       :", r2_cp)
    
    return y_pred, y_true, mae_ct, mae_cp, r2_ct, r2_cp


def save_model_pipeline(model, misc_scaler, y_scaler, N_radial, N_adv):
    """Save the complete model pipeline for inference."""
    torch.save(model.state_dict(), os.path.join(out_dir, "propnet_weights.pt"))
    joblib.dump(misc_scaler, os.path.join(out_dir, "misc_scaler.joblib"))
    joblib.dump(y_scaler, os.path.join(out_dir, "y_scaler.joblib"))
    
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({ "N_radial": N_radial, "N_adv": N_adv }, f)


def load_model_pipeline():
    """Load the complete model pipeline for inference."""
    # Load metadata
    with open(os.path.join(out_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    
    # Load scalers
    misc_scaler = joblib.load(os.path.join(out_dir, "misc_scaler.joblib"))
    y_scaler = joblib.load(os.path.join(out_dir, "y_scaler.joblib"))
    
    # Create and load model
    N_radial = meta['N_radial']
    N_adv = meta['N_adv']
    model = create_model(N_radial, N_adv)
    model.load_state_dict(torch.load(os.path.join(out_dir, "propnet_weights.pt")))
    model.eval()
    
    return model, misc_scaler, y_scaler, meta


def predict(model, misc_scaler, y_scaler, radius, chord, twist, diameter, N_blades, J_array):
    """Make predictions for given propeller geometry and operating conditions."""
    x_geom, x_misc = build_input_tensors(radius, chord, twist, diameter, N_blades, J_array)
    
    # Scale the misc features
    x_misc_scaled = misc_scaler.transform(x_misc)
    
    with torch.no_grad():
        y_hat_scaled = model(torch.tensor(x_geom), torch.tensor(x_misc_scaled)).numpy()
    
    # Inverse transform the prediction
    y_hat = y_scaler.inverse_transform(y_hat_scaled)
    
    # Reshape back into separate CT and CP arrays
    N_adv = len(J_array)
    ct_pred, cp_pred = y_hat.reshape(2, N_adv)
    return ct_pred, cp_pred


def build_input_tensors(radius, chord, twist, diameter, N_blades, J_array):
    """Build the geometry and miscellaneous tensors for a single prediction."""
    # Add a batch dimension of 1
    x_geom = np.zeros((1, 3, len(radius)), dtype=np.float32)
    x_geom[0, 0, :] = np.asarray(radius, dtype=np.float32)
    x_geom[0, 1, :] = np.asarray(chord, dtype=np.float32)
    x_geom[0, 2, :] = np.asarray(twist, dtype=np.float32)

    misc_scalars = np.array([diameter, N_blades], dtype=np.float32)
    j_array_np = np.asarray(J_array, dtype=np.float32)
    x_misc = np.concatenate([misc_scalars, j_array_np]).reshape(1, -1)
    
    return x_geom, x_misc


def main():
    """Main training pipeline."""
    print("Loading data...")
    df, _, _, _ = read_data()
    if df is None:
        return

    # First, diagnose the data for inconsistencies
    if not diagnose_data_homogeneity(df):
        print("Data is not homogeneous. Aborting training.")
        return
    
    N_adv = len(df.iloc[0]['J'])
    N_radial = len(df.iloc[0]['radius'])

    print("Preparing data...")
    (X_geom_train, X_misc_train, X_geom_val, X_misc_val, X_geom_test, X_misc_test,
     y_train, y_val, y_test, misc_scaler, y_scaler) = prepare_data(df, None, None)
    
    print("Creating data loaders...")
    train_loader = create_data_loaders(X_geom_train, X_misc_train, y_train)
    
    print("Creating model...")
    model = create_model(N_radial, N_adv)
    
    print("Training model...")
    model = train_model(model, train_loader, X_geom_val, X_misc_val, y_val)
    
    print("Evaluating model...")
    evaluate_model(model, X_geom_test, X_misc_test, y_test, y_scaler, N_adv)
    
    print("Saving model pipeline...")
    save_model_pipeline(model, misc_scaler, y_scaler, N_radial, N_adv)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

