import openvsp_config
openvsp_config.LOAD_GRAPHICS = True
openvsp_config.LOAD_FACADE = True
import openvsp as vsp
import os, pdb, time

# Optional: verify GUI build is available
if not vsp.IsGUIBuild():
    raise RuntimeError("This OpenVSP build does not include GUI support.")

# Initialize GUI, load your BEM, then start the GUI
vsp.InitGUI()
vsp.ClearVSPModel()
# find the bem file in the current directory
bem_path = os.path.join(os.path.dirname(__file__), "apc29ff_9x5_geom.bem")

prop_id = vsp.ImportFile(bem_path, vsp.IMPORT_BEM, "") 

# Optionally set BEM prop as active for BEM ops
vsp.SetBEMPropID(prop_id)

def find_parm_by_keywords(geom_id, keywords):
    """
    Find a parameter by keywords and return a dictionary with all relevant fields.
    Returns:
        dict: {
            "pid": parameter id (str),
            "name": parameter name (str),
            "group": parameter group (str),
            "display_group": parameter display group (str),
            "description": parameter description (str)
        }
        If not found, all fields are empty strings.
    """
    kw = [k.lower() for k in keywords]
    for pid in vsp.GetGeomParmIDs(geom_id):
        name = vsp.GetParmName(pid)
        group = vsp.GetParmGroupName(pid)
        disp = vsp.GetParmDisplayGroupName(pid)
        desc = vsp.GetParmDescript(pid)
        blob = " ".join([name.lower(), group.lower(), disp.lower(), desc.lower()])
        if all(k in blob for k in kw):
            return {
                "pid": pid,
                "name": name,
                "group": group,
                "display_group": disp,
                "description": desc
            }
    return {
        "pid": "",
        "name": "",
        "group": "",
        "display_group": "",
        "description": ""
    }

# Need to set feather axis to 0.125 and construct x/c to 0.0

feather_id = vsp.GetParm(prop_id, "FeatherAxisXoC", "Design")
construct_id = vsp.GetParm(prop_id, "ConstructXoC", "Design")
vsp.SetParmVal(construct_id, 0.0)

# Optional: verify
print("Feather Axis:", vsp.GetParmVal(feather_id) if feather_id else "set via fallback")
print("Construction X/C:", vsp.GetParmVal(construct_id) if construct_id else "set via fallback")

# Start GUI after setting params
if openvsp_config.LOAD_GRAPHICS:
    vsp.StartGUI()
    vsp.SetShowBorders(False)
    vsp.SetViewAxis(False)
pdb.set_trace()