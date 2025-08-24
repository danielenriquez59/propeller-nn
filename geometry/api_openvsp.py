import openvsp_config
openvsp_config.LOAD_GRAPHICS = True
openvsp_config.LOAD_FACADE = True
import openvsp as vsp
import os, pdb, time


def send_bem_to_vsp(bem_path: str):
    """Load a BEM file into OpenVSP, set prop parameters, and optionally start the GUI.

    Args:
        bem_path: Path to a .bem file compatible with OpenVSP.
    Returns:
        str: The imported propeller geometry ID.
    """
    if not vsp.IsGUIBuild():
        raise RuntimeError("This OpenVSP build does not include GUI support.")

    vsp.InitGUI()
    vsp.ClearVSPModel()

    prop_id = vsp.ImportFile(bem_path, vsp.IMPORT_BEM, "")
    vsp.SetBEMPropID(prop_id)

    # Set recommended parameters
    feather_id = vsp.GetParm(prop_id, "FeatherAxisXoC", "Design")
    construct_id = vsp.GetParm(prop_id, "ConstructXoC", "Design")
    if feather_id:
        vsp.SetParmVal(feather_id, 0.125)
    if construct_id:
        vsp.SetParmVal(construct_id, 0.0)

    # Optional: verify
    if feather_id:
        print("Feather Axis:", vsp.GetParmVal(feather_id))
    if construct_id:
        print("Construction X/C:", vsp.GetParmVal(construct_id))

    if openvsp_config.LOAD_GRAPHICS:
        vsp.StartGUI()
        vsp.SetShowBorders(False)
        vsp.SetViewAxis(False)

    return prop_id


if __name__ == "__main__":
    # Test with APC prop BEM created by create_geomety.py
    bem_path = os.path.join(os.path.dirname(__file__), "apc29ff_9x5_geom.bem")
    if not os.path.isfile(bem_path):
        raise FileNotFoundError(f"BEM file not found: {bem_path}")
    prop_id = send_bem_to_vsp(bem_path)
    print("Imported Prop ID:", prop_id)
    if openvsp_config.LOAD_GRAPHICS:
        pdb.set_trace()