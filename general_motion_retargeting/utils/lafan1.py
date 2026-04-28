import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


FORMAT_CONFIGS = {
    "lafan1": {
        "rotation_matrix": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        "left_foot_bone": "LeftToe",
        "right_foot_bone": "RightToe",
    },
    "nokov": {
        "rotation_matrix": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        "left_foot_bone": "LeftToeBase",
        "right_foot_bone": "RightToeBase",
    },
    "mocap": {
        "rotation_matrix": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        "left_foot_bone": "LeftToe",
        "right_foot_bone": "RightToe",
    },
}


def load_bvh_file(bvh_file, format="lafan1"):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    if format not in FORMAT_CONFIGS:
        raise ValueError(f"Invalid format: {format}")

    format_config = FORMAT_CONFIGS[format]

    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = format_config["rotation_matrix"]
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = [position, orientation]
            print("---------")
            print(bone)
        # Add modified foot pose. Different BVH exports may use different toe names.
        result["LeftFootMod"] = [result["LeftFoot"][0], result[format_config["left_foot_bone"]][1]]
        result["RightFootMod"] = [result["RightFoot"][0], result[format_config["right_foot_bone"]][1]]
            
        frames.append(result)
    
    # human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height

