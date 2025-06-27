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
import ast
import pdb

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
    Transforms the DataFrame into flattened numpy arrays for training.

    For each row in the DataFrame, it:
    1. Flattens the geometry arrays (radius, chord, twist) and scalar
       metadata (pitch, diameter, N_blades) along with the 'J' array
       into a single feature vector `X`.
    2. Flattens the 'CT' and 'CP' target arrays into a single target
       vector `y`.
    3. Splits the data into training, validation, and test sets.
    4. Applies StandardScaler to both features and targets.
    """
    # Infer array lengths from the first row
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot prepare data.")
    
    N_radial = len(df.iloc[0]["radius"])
    N_adv = len(df.iloc[0]["J"])
    
    input_dim = 3 * N_radial + 2 + N_adv
    output_dim = 2 * N_adv

    def build_X(row):
        geom = np.concatenate([
            np.asarray(row["radius"], dtype=np.float32),
            np.asarray(row["chord"], dtype=np.float32),
            np.asarray(row["twist"], dtype=np.float32)
        ])
        scalars = np.array([row["diameter"], row["N_blades"]], dtype=np.float32)
        j_array = np.asarray(row["J"], dtype=np.float32)
        return np.concatenate([geom, scalars, j_array])

    def build_y(row):
        return np.concatenate([
            np.asarray(row["CT"], dtype=np.float32),
            np.asarray(row["CP"], dtype=np.float32)
        ])

    X = np.vstack(df.apply(build_X, axis=1).to_numpy())
    y = np.vstack(df.apply(build_y, axis=1).to_numpy())
    
    # Assert dimensions
    assert X.shape[1] == input_dim, f"X dimension mismatch: expected {input_dim}, got {X.shape[1]}"
    assert y.shape[1] == output_dim, f"y dimension mismatch: expected {output_dim}, got {y.shape[1]}"

    # Split: 70% train, 15% val, 15% test
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)

    # Fit scalers on training data only
    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)

    # Apply scaling to all sets
    X_train = x_scaler.transform(X_train)
    X_val = x_scaler.transform(X_val)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler, input_dim, output_dim, N_adv


def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=256):
    """Create PyTorch data loaders for training and validation."""
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


class PropNet(nn.Module):
    """Neural network for propeller performance prediction."""
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


def create_model(input_dim, hidden=(256, 128, 64), output_dim=2):
    """Create and initialize the PropNet model."""
    model = PropNet(in_dim=input_dim, hidden=hidden, out_dim=output_dim)
    return model


def train_model(model, train_loader, X_val, y_val, epochs=200, lr=2e-3, weight_decay=1e-4, patience=10):
    """Train the model with early stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val, wait = np.inf, 0
    
    for epoch in range(epochs):
        # ---- train one epoch ----
        model.train()
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---- validation ----
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(torch.tensor(X_val)), torch.tensor(y_val)).item()
        
        print(f"epoch {epoch:3d} | val MSE {val_loss:.4g}")

        if val_loss < best_val:
            torch.save(model.state_dict(), "best.pt")
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
    
    return model


def evaluate_model(model, X_test, y_test, y_scaler, N_adv):
    """Evaluate the trained model on test set."""
    # Load best model weights
    model.load_state_dict(torch.load("best.pt"))
    model.eval()
    
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test)).numpy()
    
    # Invert scaling to get actual values
    y_pred = y_scaler.inverse_transform(y_pred)
    y_true = y_scaler.inverse_transform(y_test)

    # Reshape to separate CT and CP
    # Shape becomes (n_samples, 2, N_adv)
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


def save_model_pipeline(model, x_scaler, y_scaler, feature_cols, target_cols, N_radial, N_adv):
    """Save the complete model pipeline for inference."""
    torch.save(model.state_dict(), "propnet_weights.pt")
    joblib.dump(x_scaler, "x_scaler.joblib")
    joblib.dump(y_scaler, "y_scaler.joblib")
    
    with open("meta.json", "w") as f:
        json.dump({
            "N_radial": N_radial,
            "N_adv": N_adv,
            "input_order": feature_cols, # This is conceptual
            "target_order": target_cols # This is conceptual
        }, f)


def load_model_pipeline():
    """Load the complete model pipeline for inference."""
    # Load metadata
    with open("meta.json", "r") as f:
        meta = json.load(f)
    
    # Load scalers
    x_scaler = joblib.load("x_scaler.joblib")
    y_scaler = joblib.load("y_scaler.joblib")
    
    # Create and load model
    N_radial = meta['N_radial']
    N_adv = meta['N_adv']
    input_dim = 3 * N_radial + 2 + N_adv
    output_dim = 2 * N_adv
    model = create_model(input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load("propnet_weights.pt"))
    model.eval()
    
    return model, x_scaler, y_scaler, meta


def predict(model, x_scaler, y_scaler, radius, chord, twist, diameter, N_blades, J_array):
    """Make predictions for given propeller geometry and operating conditions."""
    # Build feature vector (same ordering as training)
    x = build_feature_vector(radius, chord, twist, diameter, N_blades, J_array)
    x = x_scaler.transform(x[None, :])
    
    with torch.no_grad():
        y_hat = model(torch.tensor(x)).numpy()
    
    # Inverse transform to get actual thrust and torque values
    y_hat = y_scaler.inverse_transform(y_hat)
    
    # Reshape back into separate CT and CP arrays
    N_adv = len(J_array)
    ct_pred, cp_pred = y_hat.reshape(2, N_adv)
    return ct_pred, cp_pred


def build_feature_vector(radius, chord, twist, diameter, N_blades, J_array):
    """Build feature vector from propeller geometry and operating conditions."""
    geom = np.concatenate([radius, chord, twist])
    scalars = np.array([diameter, N_blades], dtype=np.float32)
    features = np.concatenate([geom, scalars, np.asarray(J_array, dtype=np.float32)])
    return features.astype(np.float32)


def main():
    """Main training pipeline."""
    print("Loading data...")
    df, feature_cols, target_cols, N_radial = read_data()
    if df is None:
        return

    # First, diagnose the data for inconsistencies
    if not diagnose_data_homogeneity(df):
        print("Data is not homogeneous. Aborting training.")
        print("Please fix the data source or filter the DataFrame before proceeding.")
        return

    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler, input_dim, output_dim, N_adv = prepare_data(
        df, feature_cols, target_cols
    )
    
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(X_train, y_train, X_val, y_val)
    
    print("Creating model...")
    model = create_model(input_dim=input_dim, output_dim=output_dim)
    
    print("Training model...")
    model = train_model(model, train_loader, X_val, y_val)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, y_scaler, N_adv)
    
    print("Saving model pipeline...")
    save_model_pipeline(model, x_scaler, y_scaler, feature_cols, target_cols, N_radial, N_adv)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

