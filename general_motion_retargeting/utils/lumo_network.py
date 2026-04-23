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

POSITION_SCALE_METERS = 0.01
COORDINATE_ROTATION_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
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


def build_gmr_human_frame_from_lumo(
    bones_by_name: Dict[str, object],
    position_scale: float = POSITION_SCALE_METERS,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for bone_name, bone in bones_by_name.items():
        pos = np.array([bone.X, bone.Y, bone.Z], dtype=np.float64)
        quat_wxyz = normalize_quat(np.array([bone.qw, bone.qx, bone.qy, bone.qz], dtype=np.float64))

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
    }
    return frame
