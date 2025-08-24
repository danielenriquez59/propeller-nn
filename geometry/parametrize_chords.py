# an attempt to parametrize the chord distribution using Bernstein polynomials to reduce the number of parameters
# this is a work in progress, and is not currently used in the project

import numpy as np
import pdb
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
try:
    from scipy.optimize import least_squares
except ImportError:
    raise ImportError("This script requires SciPy (for least_squares). pip install scipy")
from sklearn.decomposition import PCA
from scipy.special import comb

# ---------------------------- Model family ---------------------------- #

def bernstein_model(t, a):  # a: length n
    n = len(a)
    # (1 - t) * sum a_i * C(n-1,i) t^i (1-t)^(n-1-i)
    poly = sum(a[i] * comb(n-1, i) * (t**i) * ((1-t)**(n-1-i)) for i in range(n))
    return (1 - t) * poly  # guarantees c(1)=0, free root

# ---------------------------- Fitting per curve ---------------------------- #

@dataclass
class FitResult:
    params: np.ndarray  # [A, p, q, s1, s2] (truncate if lower order)
    rmse: float
    max_abs: float
    area_err_pct: float

def fit_one_curve(t: np.ndarray,
                  c: np.ndarray,
                  poly_order: int = 2,
                  weights: Optional[np.ndarray] = None,
                  match_tip_zero: bool = True) -> FitResult:
    """
    Fit one chord curve using Bernstein basis. params are the Bernstein coefficients a (length n),
    where n = poly_order + 1. Model: c(t) = (1 - t) * sum a_i * C(n-1,i) t^i (1-t)^(n-1-i)
    """
    c = np.asarray(c).astype(float)
    # Optional: force last point to near zero (helps a lot if noisy near tip)
    if match_tip_zero:
        c = c.copy()
        c[-1] = min(c[-1], 1e-4)

    # Weights: emphasize mid–outer span where area/solidity matters more
    if weights is None:
        weights = t  # down-weight near hub

    # number of coefficients
    n = int(poly_order) + 1
    n = max(1, n)

    # Build weighted Bernstein design matrix for initial guess
    B = np.stack([comb(n - 1, i) * (t ** i) * ((1 - t) ** (n - 1 - i)) for i in range(n)], axis=1)
    M = (1.0 - t)[:, None] * B
    Mw = weights[:, None] * M
    cw = weights * c

    # Unconstrained least squares init, then clip to bounds
    try:
        a0, *_ = np.linalg.lstsq(Mw, cw, rcond=None)
    except np.linalg.LinAlgError:
        a0 = np.full(n, np.max(c))

    A0 = max(1e-6, float(np.max(c)))
    lb = np.zeros(n, dtype=float)
    ub = np.full(n, 5.0 * A0, dtype=float)
    x0 = np.clip(a0, lb, ub)

    def residuals(a):
        pred = bernstein_model(t, a)
        return weights * (pred - c)

    res = least_squares(residuals, x0, bounds=(lb, ub), method="trf", xtol=1e-10, ftol=1e-10)
    x = res.x

    # Metrics
    pred = bernstein_model(t, x)
    rmse = float(np.sqrt(np.mean((pred - c) ** 2)))
    max_abs = float(np.max(np.abs(pred - c)))
    area_err_pct = float(100.0 * (np.trapz(pred, t) - np.trapz(c, t)) / max(1e-12, np.trapz(c, t)))

    return FitResult(params=x, rmse=rmse, max_abs=max_abs, area_err_pct=area_err_pct)

# ---------------------------- Batch + PCA ---------------------------- #

@dataclass
class BatchResult:
    t: np.ndarray
    params_matrix: np.ndarray   # (n_curves, n_params)
    metrics: Dict[str, Any]     # mean/std of errors
    pca: PCA
    mean_params: np.ndarray
    components: np.ndarray      # PCA components in coefficient space
    explained_variance_ratio: np.ndarray

def fit_all_curves(chords: np.ndarray,
                   poly_order: int = 2) -> Tuple[BatchResult, np.ndarray, np.ndarray]:
    """
    chords: (n_curves, n_points), t is assumed linspace(0,1,n_points)
    returns: (BatchResult, reconstructed_from_full_params, errors_full_fit)
    """
    n_curves, n_pts = chords.shape
    t = np.linspace(0.0, 1.0, n_pts)

    # Fit each curve
    fits = [fit_one_curve(t, chords[i], poly_order=poly_order) for i in range(n_curves)]
    params = np.stack([f.params for f in fits], axis=0)

    # Reconstruct with full params (sanity check)
    recon = []
    for x in params:
        recon.append(bernstein_model(t, x))
    recon = np.stack(recon, axis=0)

    # Errors
    errs = chords - recon
    rmse = np.sqrt(np.mean(errs**2, axis=1))
    max_abs = np.max(np.abs(errs), axis=1)
    area_true = np.trapz(chords, t, axis=1)
    area_pred = np.trapz(recon,  t, axis=1)
    area_err_pct = 100.0 * (area_pred - area_true) / np.maximum(1e-12, area_true)

    metrics = {
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "max_abs_mean": float(np.mean(max_abs)),
        "area_err_pct_mean": float(np.mean(area_err_pct)),
    }

    # PCA on coefficients (centered)
    pca = PCA(n_components=None, svd_solver="full")
    pca.fit(params)
    result = BatchResult(
        t=t,
        params_matrix=params,
        metrics=metrics,
        pca=pca,
        mean_params=pca.mean_.copy(),
        components=pca.components_.copy(),
        explained_variance_ratio=pca.explained_variance_ratio_.copy()
    )
    return result, recon, np.stack([rmse, max_abs, area_err_pct], axis=1)

# ---------------------------- Reduced reconstruction ---------------------------- #

def reconstruct_from_pca(result: BatchResult,
                         params: np.ndarray,
                         n_components: int,
                         poly_order: int = 2) -> np.ndarray:
    """
    Rebuild a chord distribution from reduced PCA coordinates.
    params: original coefficient vector (so we can project to PCA space),
            or you can directly pass PCA scores (see below).
    n_components: number of PCA components to keep.
    Returns c(t) on result.t
    """
    # Project full params to PCA scores then truncate
    scores_full = (params - result.mean_params) @ result.components.T
    scores_trunc = np.zeros_like(scores_full)
    scores_trunc[:n_components] = scores_full[:n_components]
    params_recon = result.mean_params + scores_trunc @ result.components

    # number of coefficients equals PCA param dimension
    n_params = result.params_matrix.shape[1]
    a = params_recon[:n_params]
    t = result.t
    return bernstein_model(t, a)
    

# ---------------------------- Example usage ---------------------------- #
if __name__ == "__main__":
    from ..learn_apc import read_data
    import matplotlib.pyplot as plt
    df, feature_cols, target_cols, N_radial = read_data()
    chords = np.stack(df['chord'].to_numpy(), axis=0)
    # get unique chords
    chords = np.unique(chords, axis=0)
    n_curves, n_pts = chords.shape
    
    # Fit all curves and run PCA
    poly_order = 10
    result, recon, errs = fit_all_curves(chords, poly_order=poly_order)

    print("Fit metrics:", result.metrics)
    print("Explained variance ratio by component:", result.explained_variance_ratio)

    # Example: reconstruct the first curve using k PCA components
    # Do this for 5 curves
    
    for i in range(1,10,1):
        k = 3  # keep 3 components (usually enough)
        c_reduced = reconstruct_from_pca(result, result.params_matrix[i], n_components=k, poly_order=poly_order)
        rmse_reduced = np.sqrt(np.mean((c_reduced - chords[i])**2))
        print(f"RMSE (curve {i}) with {k} PCA comps:", rmse_reduced)

        # plot the curve
        plt.figure(figsize=(10, 5))
        plt.plot(result.t, chords[i], label='True')
        plt.plot(result.t, c_reduced, label='Reconstructed')
        plt.legend()
        plt.show()

        print(f"RMSE (curve {i}) with {k} PCA comps:", rmse_reduced)
    pdb.set_trace()
    
