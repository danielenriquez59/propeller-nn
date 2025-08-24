# download the UIUC Database from https://m-selig.ae.illinois.edu/props/download/UIUC-propDB.zip

uiuc_dir = 'C:/Users/denriquez/Documents/CodingDojo/apcNN/UIUC-propDB/'

import os
import pandas as pd
import numpy as np
import re, pdb

def determine_filetype(file):
    """
    Determines the type of the file by reading its first two lines.
    If either line contains 'CT', the filetype is 'performance', else 'geom'.
    """
    with open(file, 'r') as f:
        lines = [f.readline(), f.readline()]
        if any('CT' in line for line in lines):
            return 'performance'
        else:
            return 'geom'

def calculate_tip_mach(RPM, diameter_m, speed=330):
    """
    Calculate the tip Mach number for a propeller.
    """
    tip_speed = RPM * diameter_m * np.pi / 60
    mach = tip_speed / speed
   
    return mach

def main():
    # Define folders to search
    folders = ["volume-1/data", "volume-2/data", "volume-3/data", "volume-4/data"]

    # First pass: collect all performance files and build performance DataFrame
    perf_rows = []
    perf_file_info = []  # To store (folder, filename, prop_name, pitch, diameter, N_blades)
    for f in folders:
        folder_path = os.path.join(uiuc_dir, f)
        for file in os.listdir(folder_path):
            if file.endswith('.txt') and 'geom' not in file and 'static' not in file:
                file_path = os.path.join(folder_path, file)
                filetype = determine_filetype(file_path)
                if filetype != 'performance':
                    continue
                prop_name, pitch, diameter, N_blades, RPM = read_propeller_info(file)
                tip_mach = calculate_tip_mach(RPM, diameter)
                
                if pitch is None or diameter is None:
                    continue
                try:
                    J_new, CT_interp, CP_interp, eta_interp = read_performance(file_path)
                except Exception as e:
                    print(f"Error reading performance file {file_path}: {e}")
                    continue
                perf_rows.append({
                    "propeller_names": prop_name,
                    "pitch": pitch,
                    "diameter": diameter,
                    "N_blades": N_blades,
                    "performance_filename": file_path,
                    "J": J_new,
                    "CT": CT_interp,
                    "CP": CP_interp,
                    "eta": eta_interp,
                    "RPM": RPM,
                    "tip_mach": tip_mach
                })
                perf_file_info.append((prop_name, pitch, diameter, N_blades, f, file))
    performance = pd.DataFrame(perf_rows)

    # Second pass: for each row in performance, find and append geometry and geom_filename
    # We'll assume geometry file is in the same folder and has 'geom' in the name and same prop_name
    geom_cache = {}  # cache geometry by (folder, prop_name)
    geom_cols = ["radius", "chord", "twist", "geom_filename"]
    
    for idx, row in performance.iterrows():
        print(f"now parsing performance row {idx}")
        prop_name = row["propeller_names"]
        pitch = row["pitch"]
        diameter = row["diameter"]
        N_blades = row["N_blades"]
        # Find the folder from perf_file_info
        folder = None
        for info in perf_file_info:
            if (info[0], info[1], info[2], info[3]) == (prop_name, pitch, diameter, N_blades):
                folder = info[4]
                break
        if folder is None:
            # fallback: try all folders
            search_folders = folders
        else:
            search_folders = [folder]
        geom_found = False
        for f in search_folders:
            folder_path = os.path.join(uiuc_dir, f)
            for file in os.listdir(folder_path):
                if file.endswith('.txt'):
                    file_path = os.path.join(folder_path, file)
                    filetype = determine_filetype(file_path)
                    if filetype != 'geom':
                        continue
                    # Try to match prop_name
                    geom_prop_name, _, _, _, _ = read_propeller_info(file)
                    if geom_prop_name == prop_name:
                        try:
                            r_R_new, c_R_interp, beta_interp = read_geom(file_path)
                        except Exception as e:
                            print(f"Error reading geometry file {file_path}: {e}")
                            continue
                        # Add geometry info to the DataFrame
                        for col, val in zip(geom_cols, [r_R_new, c_R_interp, beta_interp, file]):
                            performance.at[idx, col] = val
                        geom_found = True
                        break
            if geom_found:
                break
        if not geom_found:
            for col in geom_cols:
                performance.at[idx, col] = None

    # Save the single combined DataFrame
    performance = performance[performance['geom_filename'].notna()]
    # Write the combined DataFrame to CSV for easy inspection
    # performance.to_csv('propeller_data_combined.csv', index=False)
    performance.to_pickle('propeller_data_combined.pkl')
    print("done processing data")
    return

def read_propeller_info(filename):
    """
    Reads the propeller info from the filename.
    Returns:
        prop_name: str
        pitch: float
        diameter (meters): float
        N_blades: int
        RPM: int
    """
    def calc_pitch(diameter, pitch_angle_deg):
        return 2*3.145*(diameter*.75)*np.tan(np.deg2rad(pitch_angle_deg))
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    if len(parts) >= 2:
        prop_name = parts[0] + '_' + parts[1]
    else:
        prop_name = base_name  # fallback if unexpected form
    # Try to extract diameter and pitch from the prop_name (e.g., ance_8.5x6_...)
    diameter = None
    pitch = None
    if len(parts) >= 2 and 'x' in parts[1]:
        dims = parts[1].split('x')
        if len(dims) == 2:
            try:
                diameter = float(dims[0])
                pitch = float(dims[1])
            except ValueError:
                diameter = None
                pitch = None
    # All propellers have 2 blades
    N_blades = 2
    blade_3 = ['mit_4.3x', 'mit_5x4']
    if any(blade in filename for blade in blade_3):
        N_blades = 3

    RPM = None
    if len(parts) > 3:
        try:
            RPM = int(parts[-1])
        except:
            RPM = None
    
    if diameter is None:
        if 'tri' in parts[1]:
            diameter = 2.55
            pitch = calc_pitch(diameter, 20)
        elif 'u80' in parts[1]:
            diameter = 3.15
            pitch = calc_pitch(diameter, 15)
        else:
            try:
                diameter = float(parts[1])
                pitch = float(parts[2].replace('deg', ''))
            except:
                diameter = None
                pitch = None
    
    # convert diameter to meters
    if diameter is not None:
        if diameter < 20:
            # in
            diameter = diameter * 0.0254
            pitch = pitch * 0.0254
        else:
            # mm
            diameter = diameter * .001
            pitch = pitch * .001

    return prop_name, pitch, diameter, N_blades, RPM

def read_geom(file):
    # Read the geometry file with three columns: r/R, c/R, beta
    try:
        data = pd.read_csv(
        file,
        sep=r'\s+',
        header=0,  # Use the first row as header
        )
        # Ensure all columns are floats (except possibly the header row)
        data = data.astype(float)
    except:
        try: 
            data = pd.read_csv(
            file,
            sep=r'\s+',
            header=1,  # Use the 2nd row as header
            )
            # Ensure all columns are floats (except possibly the header row)
            data = data.astype(float)
        except:
            pdb.set_trace()
    # Interpolate c_R and beta to 20 evenly spaced r_R points between min and max r_R
    r_R_new = np.linspace(data['r/R'].min(), data['r/R'].max(), 20)

    # Interpolate c_R
    c_R_interp = np.interp(r_R_new, data['r/R'], data['c/R'])

    # Interpolate beta
    beta_interp = np.interp(r_R_new, data['r/R'], data['beta'])


    return r_R_new, c_R_interp, beta_interp

def read_performance(file):
    # Read the performance file with four columns: J, CT, CP, eta
    try:
        data = pd.read_csv(
        file,
        sep=r'\s+',
        header=0,  # Use the first row as header
        )
        # Ensure all columns are floats (except possibly the header row)
        data = data.astype(float)
    except:
        try: 
            data = pd.read_csv(
            file,
            sep=r'\s+',
            header=1,  # Use the 2nd row as header
            )
            # Ensure all columns are floats (except possibly the header row)
            data = data.astype(float)
        except:
            pdb.set_trace()
    
    J_new = np.linspace(data['J'].min(), data['J'].max(), 20)

    # Interpolate so the data has 20 points
    CT_interp = np.interp(J_new, data['J'], data['CT'])
    CP_interp = np.interp(J_new, data['J'], data['CP'])
    eta_interp = np.interp(J_new, data['J'], data['eta'])

    return J_new, CT_interp, CP_interp, eta_interp

if __name__ == "__main__":
    main()