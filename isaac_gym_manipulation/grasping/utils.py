import xml.etree.ElementTree as ET
import os


def get_ycb_mesh_path_from_urdf(asset_root, urdf_path):
    root = ET.parse(os.path.join(asset_root, urdf_path)).getroot()
    collision_nodes = root.findall("link/collision/geometry/mesh")
    assert len(collision_nodes), "loaded object has multiple collision mesh"
    rel_col_mesh_math = collision_nodes[0].get("filename")
    full_path = os.path.join(asset_root, os.path.split(urdf_path)[0], rel_col_mesh_math)

    return full_path
