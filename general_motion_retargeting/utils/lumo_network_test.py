from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils


STREAM_BONE_ORDER: List[str] = [
    "Hips",
    "Spine2",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFootMod",
    "RightFootMod",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
]

WHOLE_BODY_BONES: List[str] = [
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToe",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToe",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]

SMPL24_POSITION_JOINTS: List[str] = WHOLE_BODY_BONES + ["LeftFinger", "RightFinger"]
FINGER_X_OFFSET_METERS = 0.10

LUMO_REQUIRED_SOURCE_BONES: List[str] = [
    "Hips",
    "Spine2",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFoot",
    "RightFoot",
    "LeftToe",
    "RightToe",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
]

LUMO_PARENT_BY_BONE = {
    "Root": None,
    "Hips": "Root",
    "LeftUpLeg": "Hips",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "LeftToe": "LeftFoot",
    "LToeEnd": "LeftToe",
    "RightUpLeg": "Hips",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
    "RightToe": "RightFoot",
    "RToeEnd": "RightToe",
    "Spine1": "Hips",
    "Spine2": "Spine1",
    "Chest": "Spine2",
    "Neck": "Chest",
    "Head": "Neck",
    "HeadEnd": "Head",
    "LeftShoulder": "Chest",
    "LeftArm": "LeftShoulder",
    "LeftForeArm": "LeftArm",
    "LeftHand": "LeftForeArm",
    "RightShoulder": "Chest",
    "RightArm": "RightShoulder",
    "RightForeArm": "RightArm",
    "RightHand": "RightForeArm",
}

POSITION_SCALE_METERS = 0.001
COORDINATE_ROTATION_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
# COORDINATE_ROTATION_MATRIX = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
COORDINATE_ROTATION_QUAT_WXYZ = np.array(
    [
        R.from_matrix(COORDINATE_ROTATION_MATRIX).as_quat()[3],
        R.from_matrix(COORDINATE_ROTATION_MATRIX).as_quat()[0],
        R.from_matrix(COORDINATE_ROTATION_MATRIX).as_quat()[1],
        R.from_matrix(COORDINATE_ROTATION_MATRIX).as_quat()[2],
    ],
    dtype=np.float64,
)

FLOATS_PER_BONE = 7
FRAME_FLOAT_COUNT = len(STREAM_BONE_ORDER) * FLOATS_PER_BONE
FRAME_BYTE_SIZE = FRAME_FLOAT_COUNT * 4

WHOLE_BODY_FLOAT_COUNT = len(WHOLE_BODY_BONES) * FLOATS_PER_BONE
WHOLE_BODY_BYTE_SIZE = WHOLE_BODY_FLOAT_COUNT * 4


def normalize_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat_wxyz)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat_wxyz / norm


def select_tracked_skeleton(frame):
    for skeleton in frame.skeletons:
        if skeleton.IsTrack:
            return skeleton
    return None


def build_bone_map(skeleton) -> Dict[str, object]:
    return {bone.Name: bone for bone in skeleton.skeletonBones}


def compute_lumo_global_poses(
    bones_by_name: Dict[str, object],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def compute(bone_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if bone_name in cache:
            return cache[bone_name]

        bone = bones_by_name[bone_name]
        local_pos = np.array([bone.X, bone.Y, bone.Z], dtype=np.float64)
        local_quat = normalize_quat(np.array([bone.qw, bone.qx, bone.qy, bone.qz], dtype=np.float64))
        parent_name = LUMO_PARENT_BY_BONE.get(bone_name)

        if parent_name is not None and parent_name in bones_by_name:
            parent_pos, parent_quat = compute(parent_name)
            global_pos = parent_pos + lafan_utils.quat_mul_vec(
                parent_quat[np.newaxis, :],
                local_pos[np.newaxis, :],
            )[0]
            global_quat = normalize_quat(
                lafan_utils.quat_mul(parent_quat[np.newaxis, :], local_quat[np.newaxis, :])[0]
            )
        else:
            global_pos = local_pos
            global_quat = local_quat

        cache[bone_name] = (global_pos, global_quat)
        return cache[bone_name]

    return {bone_name: compute(bone_name) for bone_name in bones_by_name}


def build_gmr_human_frame_from_lumo(
    bones_by_name: Dict[str, object],
    position_scale: float = POSITION_SCALE_METERS,
    lumo_positions_are_local: bool = True,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    source_poses = (
        compute_lumo_global_poses(bones_by_name)
        if lumo_positions_are_local
        else {
            bone_name: (
                np.array([bone.X, bone.Y, bone.Z], dtype=np.float64),
                normalize_quat(np.array([bone.qw, bone.qx, bone.qy, bone.qz], dtype=np.float64)),
            )
            for bone_name, bone in bones_by_name.items()
        }
    )

    for bone_name, (pos, quat_wxyz) in source_poses.items():
        transformed_pos = pos @ COORDINATE_ROTATION_MATRIX.T * position_scale
        transformed_quat = normalize_quat(
            lafan_utils.quat_mul(
                COORDINATE_ROTATION_QUAT_WXYZ[np.newaxis, :], quat_wxyz[np.newaxis, :]
            )[0]
        )
        result[bone_name] = [transformed_pos, transformed_quat]

    if "LeftFoot" in result and "LeftToe" in result:
        result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
    if "RightFoot" in result and "RightToe" in result:
        result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]

    return result


def missing_stream_bones(human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[str]:
    return [bone_name for bone_name in STREAM_BONE_ORDER if bone_name not in human_frame]


def human_frame_to_array(
    human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
    dtype=np.float32,
) -> np.ndarray:
    missing = missing_stream_bones(human_frame)
    if missing:
        raise ValueError(f"Missing required stream bones: {missing}")

    values = []
    for bone_name in STREAM_BONE_ORDER:
        pos, quat = human_frame[bone_name]
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat = normalize_quat(np.asarray(quat, dtype=np.float64).reshape(4))
        values.extend(pos.tolist())
        values.extend(quat.tolist())
    return np.asarray(values, dtype=dtype)


def array_to_human_frame(array: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.size != FRAME_FLOAT_COUNT:
        raise ValueError(f"Invalid frame float count: expected {FRAME_FLOAT_COUNT}, got {flat.size}")

    reshaped = flat.reshape(len(STREAM_BONE_ORDER), FLOATS_PER_BONE)
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for index, bone_name in enumerate(STREAM_BONE_ORDER):
        row = reshaped[index]
        result[bone_name] = [
            row[:3].astype(np.float64, copy=True),
            normalize_quat(row[3:7].astype(np.float64, copy=True)),
        ]
    return result


def bytes_to_human_frame(buffer: bytes) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if len(buffer) != FRAME_BYTE_SIZE:
        raise ValueError(f"Invalid frame byte size: expected {FRAME_BYTE_SIZE}, got {len(buffer)}")
    return array_to_human_frame(np.frombuffer(buffer, dtype=np.float32))


def human_frame_to_bytes(human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> bytes:
    return human_frame_to_array(human_frame).tobytes()


def missing_whole_body_bones(human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> List[str]:
    return [bone_name for bone_name in WHOLE_BODY_BONES if bone_name not in human_frame]


def whole_body_frame_to_array(
    human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
    dtype=np.float32,
) -> np.ndarray:
    missing = missing_whole_body_bones(human_frame)
    if missing:
        raise ValueError(f"Missing required whole-body bones: {missing}")

    values = []
    for bone_name in WHOLE_BODY_BONES:
        pos, quat = human_frame[bone_name]
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        quat = normalize_quat(np.asarray(quat, dtype=np.float64).reshape(4))
        values.extend(pos.tolist())
        values.extend(quat.tolist())
    return np.asarray(values, dtype=dtype)


def array_to_whole_body_frame(array: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.size != WHOLE_BODY_FLOAT_COUNT:
        raise ValueError(f"Invalid whole-body float count: expected {WHOLE_BODY_FLOAT_COUNT}, got {flat.size}")

    reshaped = flat.reshape(len(WHOLE_BODY_BONES), FLOATS_PER_BONE)
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for index, bone_name in enumerate(WHOLE_BODY_BONES):
        row = reshaped[index]
        result[bone_name] = [
            row[:3].astype(np.float64, copy=True),
            normalize_quat(row[3:7].astype(np.float64, copy=True)),
        ]
    return result


def bytes_to_whole_body_frame(buffer: bytes) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if len(buffer) != WHOLE_BODY_BYTE_SIZE:
        raise ValueError(f"Invalid whole-body frame byte size: expected {WHOLE_BODY_BYTE_SIZE}, got {len(buffer)}")
    return array_to_whole_body_frame(np.frombuffer(buffer, dtype=np.float32))


def whole_body_frame_to_bytes(human_frame: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> bytes:
    return whole_body_frame_to_array(human_frame).tobytes()


def whole_body_frame_to_human_frame(
    whole_body_frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Derive the 14-bone IK stream dict from a decoded whole-body frame.

    All STREAM_BONE_ORDER bones are present in WHOLE_BODY_BONES except
    LeftFootMod and RightFootMod, which are reconstructed the same way
    build_gmr_human_frame_from_lumo computes them: position from Foot,
    orientation from Toe.
    """
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for bone_name in STREAM_BONE_ORDER:
        if bone_name in whole_body_frame:
            result[bone_name] = whole_body_frame[bone_name]
        elif bone_name == "LeftFootMod":
            result["LeftFootMod"] = [whole_body_frame["LeftFoot"][0], whole_body_frame["LeftToe"][1]]
        elif bone_name == "RightFootMod":
            result["RightFootMod"] = [whole_body_frame["RightFoot"][0], whole_body_frame["RightToe"][1]]
        else:
            raise KeyError(f"Cannot derive stream bone '{bone_name}' from whole-body frame")
    return result


def whole_body_frame_to_smpl24_position_dict(
    whole_body_frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
    finger_x_offset_meters: float = FINGER_X_OFFSET_METERS,
    quat_is_y_up: bool = True,
) -> Dict[str, np.ndarray]:
    """Build a 24-joint position-only dict from a decoded whole-body frame.

    The two extra joints are synthetic fingertip proxies placed +x in each
    hand's local frame, then transformed into global space.

    If quat_is_y_up is True, reconstructed global hand quaternions are
    interpreted in y-up coordinates and converted with
    COORDINATE_ROTATION_QUAT_WXYZ before applying the offset.
    """
    missing = [bone_name for bone_name in WHOLE_BODY_BONES if bone_name not in whole_body_frame]
    if missing:
        raise KeyError(f"Missing required whole-body joints for SMPL24 positions: {missing}")

    result: Dict[str, np.ndarray] = {}
    for bone_name in WHOLE_BODY_BONES:
        pos, _ = whole_body_frame[bone_name]
        result[bone_name] = np.asarray(pos, dtype=np.float64).reshape(3).copy()

    quat_global_cache: Dict[str, np.ndarray] = {}

    def get_global_quat(bone_name: str) -> np.ndarray:
        if bone_name in quat_global_cache:
            return quat_global_cache[bone_name]

        _, local_quat = whole_body_frame[bone_name]
        local_quat = normalize_quat(np.asarray(local_quat, dtype=np.float64).reshape(4))
        parent_name = LUMO_PARENT_BY_BONE.get(bone_name)

        if parent_name is not None and parent_name in whole_body_frame:
            parent_global = get_global_quat(parent_name)
            global_quat = normalize_quat(
                lafan_utils.quat_mul(parent_global[np.newaxis, :], local_quat[np.newaxis, :])[0]
            )
        else:
            global_quat = local_quat

        quat_global_cache[bone_name] = global_quat
        return global_quat

    finger_offset_local = np.array([finger_x_offset_meters, 0.0, 0.0], dtype=np.float64)
    left_hand_quat_global = get_global_quat("LeftHand")
    right_hand_quat_global = get_global_quat("RightHand")

    if quat_is_y_up:
        left_hand_quat_global = normalize_quat(
            lafan_utils.quat_mul(
                COORDINATE_ROTATION_QUAT_WXYZ[np.newaxis, :], left_hand_quat_global[np.newaxis, :]
            )[0]
        )
        right_hand_quat_global = normalize_quat(
            lafan_utils.quat_mul(
                COORDINATE_ROTATION_QUAT_WXYZ[np.newaxis, :], right_hand_quat_global[np.newaxis, :]
            )[0]
        )

    left_offset_global = lafan_utils.quat_mul_vec(
        left_hand_quat_global[np.newaxis, :], finger_offset_local[np.newaxis, :]
    )[0]
    right_offset_global = lafan_utils.quat_mul_vec(
        right_hand_quat_global[np.newaxis, :], finger_offset_local[np.newaxis, :]
    )[0]

    result["LeftFinger"] = result["LeftHand"] + left_offset_global
    result["RightFinger"] = result["RightHand"] + right_offset_global
    return result


def recv_exact(sock, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving frame payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def make_demo_human_frame(frame_index: int, fps: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    t = frame_index / max(fps, 1.0)
    root_height = 0.92 + 0.02 * np.sin(2.0 * np.pi * 0.5 * t)
    arm_swing = np.deg2rad(20.0 * np.sin(2.0 * np.pi * 0.7 * t))
    leg_swing = np.deg2rad(18.0 * np.sin(2.0 * np.pi * 0.7 * t))

    def quat_from_euler_xyz(x=0.0, y=0.0, z=0.0) -> np.ndarray:
        quat_xyzw = R.from_euler("xyz", [x, y, z], degrees=False).as_quat()
        return normalize_quat(np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64))

    frame = {
        "Hips": [np.array([0.0, 0.0, root_height]), quat_from_euler_xyz()],
        "Spine2": [np.array([0.0, 0.0, root_height + 0.25]), quat_from_euler_xyz()],
        "LeftUpLeg": [np.array([0.0, 0.08, root_height - 0.10]), quat_from_euler_xyz(x=-leg_swing)],
        "RightUpLeg": [np.array([0.0, -0.08, root_height - 0.10]), quat_from_euler_xyz(x=leg_swing)],
        "LeftLeg": [np.array([0.0, 0.08, root_height - 0.45]), quat_from_euler_xyz(x=0.5 * leg_swing)],
        "RightLeg": [np.array([0.0, -0.08, root_height - 0.45]), quat_from_euler_xyz(x=-0.5 * leg_swing)],
        "LeftFootMod": [np.array([0.05, 0.08, 0.05]), quat_from_euler_xyz()],
        "RightFootMod": [np.array([0.05, -0.08, 0.05]), quat_from_euler_xyz()],
        "LeftArm": [np.array([0.0, 0.22, root_height + 0.20]), quat_from_euler_xyz(y=-arm_swing)],
        "RightArm": [np.array([0.0, -0.22, root_height + 0.20]), quat_from_euler_xyz(y=arm_swing)],
        "LeftForeArm": [np.array([0.0, 0.42, root_height + 0.16]), quat_from_euler_xyz(y=-0.5 * arm_swing)],
        "RightForeArm": [np.array([0.0, -0.42, root_height + 0.16]), quat_from_euler_xyz(y=0.5 * arm_swing)],
        "LeftHand": [np.array([0.0, 0.58, root_height + 0.14]), quat_from_euler_xyz()],
        "RightHand": [np.array([0.0, -0.58, root_height + 0.14]), quat_from_euler_xyz()],
        "LeftFoot": [np.array([0.05, 0.08, 0.05]), quat_from_euler_xyz()],
        "LeftToe": [np.array([0.15, 0.08, 0.02]), quat_from_euler_xyz()],
        "RightFoot": [np.array([0.05, -0.08, 0.05]), quat_from_euler_xyz()],
        "RightToe": [np.array([0.15, -0.08, 0.02]), quat_from_euler_xyz()],
        "Spine1": [np.array([0.0, 0.0, root_height + 0.12]), quat_from_euler_xyz()],
        "Chest": [np.array([0.0, 0.0, root_height + 0.38]), quat_from_euler_xyz()],
        "Neck": [np.array([0.0, 0.0, root_height + 0.50]), quat_from_euler_xyz()],
        "Head": [np.array([0.0, 0.0, root_height + 0.66]), quat_from_euler_xyz()],
        "LeftShoulder": [np.array([0.0, 0.12, root_height + 0.38]), quat_from_euler_xyz()],
        "RightShoulder": [np.array([0.0, -0.12, root_height + 0.38]), quat_from_euler_xyz()],
    }
    return frame
