from learn_apc import read_data
import numpy as np
import pandas as pd
import pdb
import json

def chord_envelope(df):
    """
    Stack the chords and find the minimum and maximum values for each radius.
    """
    chords_stacked = np.stack(df["chord"].to_numpy())
    chord_min = np.min(chords_stacked, axis=0)
    chord_max = np.max(chords_stacked, axis=0)
    return chord_min, chord_max

def twist_envelope(df):
    """
    Stack the twists and find the minimum and maximum values for each radius.
    """
    twists_stacked = np.stack(df["twist"].to_numpy())
    twist_min = np.min(twists_stacked, axis=0)
    twist_max = np.max(twists_stacked, axis=0)
    return twist_min, twist_max

if __name__ == "__main__":
    df = read_data()
    chord_min, chord_max = chord_envelope(df)
    twist_min, twist_max = twist_envelope(df)
    # write a json with these 4 keys
    with open("chord_envelope.json", "w") as f:
        json.dump({"chord_min": chord_min.tolist(), "chord_max": chord_max.tolist(), "twist_min": twist_min.tolist(), "twist_max": twist_max.tolist()}, f)
    print("done")