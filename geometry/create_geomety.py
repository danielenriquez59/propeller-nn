#!/usr/bin/env python3
"""
Create a VSP BEM file from arbitrary chord and twist distributions.

This module exposes a pure function that accepts NumPy arrays directly and
writes a BEM file in the same format as `convert_PE0_to_BEM.py`.

Inputs:
- radius_R: section radii normalized by prop radius (0..1]
- chord: chord distribution (absolute or relative depending on flag)
- twist_deg: twist distribution in degrees
- diameter: prop diameter (same units as absolute chord if chord_is_relative=False)
- num_blade: number of blades

Optional inputs:
- chord_is_relative: whether chord is specified as chord/R (default True)
- t_c: thickness ratio distribution (scalar or array). Default 0.12
- rake_R, skew_R, sweep_deg: optional arrays (default zeros)
- feather, pre_cone: scalar angles in degrees
- center, normal: 3-tuples for pose

Returns the BEM matrix as np.ndarray.
"""

from typing import Optional, Sequence, Tuple
import numpy as np


def linear_interp(x: Sequence[float], y: Sequence[float], xq: float) -> float:
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    order = np.argsort(x_arr)
    return float(np.interp(xq, x_arr[order], y_arr[order]))


def write_matrix(io, mat: np.ndarray) -> None:
    rows, cols = mat.shape
    for i in range(rows):
        line = ", ".join(f"{mat[i, j]:.8f}" for j in range(cols))
        io.write(line + "\n")


def make_bem_matrix(
    radius_R: np.ndarray,
    chord_R: np.ndarray,
    twist_deg: np.ndarray,
    t_c: np.ndarray,
    rake_R: Optional[np.ndarray] = None,
    skew_R: Optional[np.ndarray] = None,
    sweep_deg: Optional[np.ndarray] = None,
) -> np.ndarray:
    n = len(radius_R)
    if rake_R is None:
        rake_R = np.zeros(n)
    if skew_R is None:
        skew_R = np.zeros(n)
    if sweep_deg is None:
        sweep_deg = np.zeros(n)
    CLi = np.zeros(n)
    axial = np.zeros(n)
    tangential = np.zeros(n)

    bem = np.column_stack(
        [
            radius_R,
            chord_R,
            twist_deg,
            rake_R,
            skew_R,
            sweep_deg,
            t_c,
            CLi,
            axial,
            tangential,
        ]
    )
    return bem


def normalize_and_validate(
    radius_R: np.ndarray,
    chord: np.ndarray,
    twist_deg: np.ndarray,
    diameter: float,
    chord_is_relative: bool,
    t_c: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not (len(radius_R) == len(chord) == len(twist_deg)):
        raise ValueError("radius, chord, and twist arrays must be the same length")
    if len(radius_R) < 2:
        raise ValueError("at least two sections are required")

    radius_R = np.asarray(radius_R, dtype=float)
    chord = np.asarray(chord, dtype=float)
    twist_deg = np.asarray(twist_deg, dtype=float)

    # Ensure strictly increasing radii for interpolation stability
    order = np.argsort(radius_R)
    radius_R = radius_R[order]
    chord = chord[order]
    twist_deg = twist_deg[order]

    # Basic validation
    if np.any(radius_R <= 0.0) or np.any(radius_R > 1.2):
        raise ValueError("radius_R values should be in (0, 1.2]; expected fractions of radius")

    R = diameter / 2.0
    chord_R = chord if chord_is_relative else chord / R

    if t_c is None:
        t_c = np.full_like(radius_R, 0.12, dtype=float)
    else:
        t_c = np.asarray(t_c, dtype=float)
        if t_c.size == 1:
            t_c = np.full_like(radius_R, float(t_c), dtype=float)
        if len(t_c) != len(radius_R):
            raise ValueError("t_c must be scalar or same length as radius")

    return radius_R, chord_R, twist_deg, t_c


def create_bem_file(
    *,
    radius_R: np.ndarray,
    chord: np.ndarray,
    twist_deg: np.ndarray,
    diameter: float,
    num_blade: int,
    output_path: str,
    chord_is_relative: bool = True,
    t_c: Optional[np.ndarray] = None,
    rake_R: Optional[np.ndarray] = None,
    skew_R: Optional[np.ndarray] = None,
    sweep_deg: Optional[np.ndarray] = None,
    feather: float = 0.0,
    pre_cone: float = 0.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: Tuple[float, float, float] = (-1.0, 0.0, 0.0),
) -> np.ndarray:
    """Create and write a BEM file from provided arrays; return the BEM matrix.

    Parameters are arrays of equal length for radius (normalized), chord, twist, and
    optional thickness ratio and geometry. Chord can be absolute or relative.
    """
    r_R, c_R, twist, t_c_arr = normalize_and_validate(
        radius_R=np.asarray(radius_R, dtype=float),
        chord=np.asarray(chord, dtype=float),
        twist_deg=np.asarray(twist_deg, dtype=float),
        diameter=float(diameter),
        chord_is_relative=bool(chord_is_relative),
        t_c=None if t_c is None else np.asarray(t_c, dtype=float),
    )

    beta_three_quarter = linear_interp(r_R, twist, 0.75)
    bem = make_bem_matrix(r_R, c_R, twist, t_c_arr, rake_R=rake_R, skew_R=skew_R, sweep_deg=sweep_deg)

    with open(output_path, "w", encoding="utf-8") as io:
        io.write("...BEM Propeller...\n")
        io.write(f"Num_Sections: {len(r_R):d}\n")
        io.write(f"Num_Blade: {int(num_blade):d}\n")
        io.write(f"Diameter: {float(diameter):.8f}\n")
        io.write(f"Beta 3/4 (deg): {beta_three_quarter:.8f}\n")
        io.write(f"Feather (deg): {float(feather):.8f}\n")
        io.write(f"Pre_Cone (deg): {float(pre_cone):.8f}\n")
        io.write(f"Center: {center[0]:.8f}, {center[1]:.8f}, {center[2]:.8f}\n")
        io.write(f"Normal: {normal[0]:.8f}, {normal[1]:.8f}, {normal[2]:.8f}\n")
        io.write("\n")
        io.write("Radius/R, Chord/R, Twist (deg), Rake/R, Skew/R, Sweep, t/c, CLi, Axial, Tangential\n")
        write_matrix(io, bem)
    print(f"BEM file written to {output_path}")
    print("After importing the BEM file into OpenVSP, set the following Propeller Design properties:")
    print("Construction X/C 0.000")
    print("Feather Axis 0.125")
    
    return bem

if __name__ == "__main__":
    # test the function
    stations = 15
    min_radius = 0.15
    max_radius = 1.0
    radius_R = np.linspace(min_radius, max_radius, stations)
    chord = np.array([0.160, 0.146, 0.144, 0.143, 0.143, 0.146, 0.151, 0.155, 0.158, 0.160, 0.159, 0.155, 0.146, 0.133, 0.114, 0.089, 0.056, 0.022])
    chord = np.interp(radius_R, np.linspace(min_radius, max_radius, len(chord)), chord) # resample chord to match radius_R

    twist_deg = np.array([31.68, 34.45, 35.93, 33.33, 29.42, 26.25, 23.67, 21.65, 20.02, 18.49, 17.06, 15.95, 14.87, 13.82, 12.77, 11.47, 10.15, 8.82])
    twist_deg = np.interp(radius_R, np.linspace(min_radius, max_radius, len(twist_deg)), twist_deg)

    diameter = 5 * 0.0254 # convert to meters
    num_blade = 2
    output_path = "apc29ff_9x5_geom.bem"
    create_bem_file(radius_R=radius_R, chord=chord, twist_deg=twist_deg, diameter=diameter, num_blade=num_blade, output_path=output_path)