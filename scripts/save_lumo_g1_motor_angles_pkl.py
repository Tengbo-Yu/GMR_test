#!/usr/bin/env python3
"""Capture LuMo G1 motor angles and save them in GMR pickle format."""

from __future__ import annotations

import argparse
import pickle
import re
import signal
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
LUMO_SDK_DIR = REPO_ROOT / "LuMoSDKPy"
DEFAULT_ROBOT_XML = REPO_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LUMO_SDK_DIR) not in sys.path:
    sys.path.insert(0, str(LUMO_SDK_DIR))

import LuMoSDKClient

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError(
        "This script now saves GMR-style pickle files and requires numpy."
    ) from exc


g_running = True


def signal_handler(signum, frame) -> None:
    del frame
    global g_running
    print(f"\nReceived signal {signum}, stopping capture...")
    g_running = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save LuMo G1 motion as a GMR-style pickle file."
    )
    parser.add_argument("--ip", type=str, default="192.168.50.150", help="LuMo SDK source IP.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "test_lumo.pkl"),
        help="Output pickle path.",
    )
    parser.add_argument(
        "--robot-name",
        type=str,
        default="UnitreeG1",
        help="Only save skeletons whose RobotName matches this value. Empty means first tracked skeleton.",
    )
    parser.add_argument(
        "--robot-xml",
        type=str,
        default=str(DEFAULT_ROBOT_XML),
        help="MuJoCo robot xml used to define the dof order.",
    )
    parser.add_argument(
        "--motion-fps",
        type=int,
        default=120,
        help="fps field written into the pickle.",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        default=False,
        help="Use non-blocking LuMo receive mode.",
    )
    parser.add_argument(
        "--poll-sleep",
        type=float,
        default=0.001,
        help="Sleep duration in seconds when non-blocking mode has no frame.",
    )
    parser.add_argument(
        "--wait-log-interval",
        type=float,
        default=2.0,
        help="How often to print waiting logs while no valid frame is available.",
    )
    return parser


def normalize_joint_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def load_joint_order(robot_xml_path: Path) -> List[str]:
    tree = ET.parse(robot_xml_path)
    joint_order = []
    for motor in tree.findall(".//actuator/motor"):
        motor_name = motor.get("name")
        if motor_name:
            joint_order.append(motor_name)
    if not joint_order:
        raise RuntimeError(f"Failed to load actuator order from {robot_xml_path}")
    return joint_order


def build_alias_to_joint_name(joint_order: List[str]) -> Dict[str, str]:
    alias_to_joint_name: Dict[str, str] = {}
    manual_aliases = {
        "left_knee_joint": ["left_knee", "left_knee_pitch", "left_leg_knee"],
        "right_knee_joint": ["right_knee", "right_knee_pitch", "right_leg_knee"],
        "left_elbow_joint": ["left_elbow", "left_elbow_pitch"],
        "right_elbow_joint": ["right_elbow", "right_elbow_pitch"],
    }

    for joint_name in joint_order:
        aliases = {joint_name}
        if joint_name.endswith("_joint"):
            aliases.add(joint_name[: -len("_joint")])
        aliases.update(manual_aliases.get(joint_name, []))
        for alias in aliases:
            alias_to_joint_name[normalize_joint_name(alias)] = joint_name

    lumo_aliases = {
        "L_LEG_HIP_PITCH": "left_hip_pitch_joint",
        "L_LEG_HIP_ROLL": "left_hip_roll_joint",
        "L_LEG_HIP_YAW": "left_hip_yaw_joint",
        "L_LEG_KNEE": "left_knee_joint",
        "L_LEG_ANKLE_PITCH": "left_ankle_pitch_joint",
        "L_LEG_ANKLE_ROLL": "left_ankle_roll_joint",
        "R_LEG_HIP_PITCH": "right_hip_pitch_joint",
        "R_LEG_HIP_ROLL": "right_hip_roll_joint",
        "R_LEG_HIP_YAW": "right_hip_yaw_joint",
        "R_LEG_KNEE": "right_knee_joint",
        "R_LEG_ANKLE_PITCH": "right_ankle_pitch_joint",
        "R_LEG_ANKLE_ROLL": "right_ankle_roll_joint",
        "WAIST_YAW": "waist_yaw_joint",
        "WAIST_ROLL": "waist_roll_joint",
        "WAIST_PITCH": "waist_pitch_joint",
        "L_SHOULDER_PITCH": "left_shoulder_pitch_joint",
        "L_SHOULDER_ROLL": "left_shoulder_roll_joint",
        "L_SHOULDER_YAW": "left_shoulder_yaw_joint",
        "L_ELBOW": "left_elbow_joint",
        "L_WRIST_ROLL": "left_wrist_roll_joint",
        "L_WRIST_PITCH": "left_wrist_pitch_joint",
        "L_WRIST_YAW": "left_wrist_yaw_joint",
        "R_SHOULDER_PITCH": "right_shoulder_pitch_joint",
        "R_SHOULDER_ROLL": "right_shoulder_roll_joint",
        "R_SHOULDER_YAW": "right_shoulder_yaw_joint",
        "R_ELBOW": "right_elbow_joint",
        "R_WRIST_ROLL": "right_wrist_roll_joint",
        "R_WRIST_PITCH": "right_wrist_pitch_joint",
        "R_WRIST_YAW": "right_wrist_yaw_joint",
    }
    for raw_name, joint_name in lumo_aliases.items():
        alias_to_joint_name[normalize_joint_name(raw_name)] = joint_name

    return alias_to_joint_name


def select_tracked_skeleton(frame, robot_name: str):
    for skeleton in frame.skeletons:
        if not skeleton.IsTrack:
            continue
        if robot_name and skeleton.RobotName != robot_name:
            continue
        return skeleton
    return None


def find_root_bone(skeleton):
    bones_by_name = {bone.Name: bone for bone in skeleton.skeletonBones}
    for candidate in ("Hips", "Pelvis", "pelvis", "Root", "Hip"):
        if candidate in bones_by_name:
            return bones_by_name[candidate]
    if skeleton.skeletonBones:
        return skeleton.skeletonBones[0]
    return None


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def transform_lumo_position(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, -z, y], dtype=np.float64) * 0.01


def transform_lumo_quat_to_xyzw(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    lumo_to_gmr_wxyz = np.array(
        [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
        dtype=np.float64,
    )
    quat_wxyz = normalize_quat_wxyz(np.array([qw, qx, qy, qz], dtype=np.float64))
    transformed_wxyz = normalize_quat_wxyz(quat_mul_wxyz(lumo_to_gmr_wxyz, quat_wxyz))
    return transformed_wxyz[[1, 2, 3, 0]]


def build_motion_frame(skeleton, joint_order: List[str], alias_to_joint_name: Dict[str, str]):
    raw_motor_angles = {str(key): float(value) for key, value in skeleton.MotorAngle.items()}
    resolved_motor_angles: Dict[str, float] = {}
    unknown_motor_names = []

    for raw_name, value in raw_motor_angles.items():
        canonical_name = alias_to_joint_name.get(normalize_joint_name(raw_name))
        if canonical_name is None:
            unknown_motor_names.append(raw_name)
            continue
        resolved_motor_angles[canonical_name] = value

    dof_pos = np.zeros(len(joint_order), dtype=np.float64)
    missing_joint_names = []
    for index, joint_name in enumerate(joint_order):
        if joint_name in resolved_motor_angles:
            dof_pos[index] = resolved_motor_angles[joint_name]
        else:
            missing_joint_names.append(joint_name)

    root_pos = np.zeros(3, dtype=np.float64)
    root_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    root_bone = find_root_bone(skeleton)
    if root_bone is not None:
        root_pos = transform_lumo_position(root_bone.X, root_bone.Y, root_bone.Z)
        root_rot = transform_lumo_quat_to_xyzw(
            root_bone.qw, root_bone.qx, root_bone.qy, root_bone.qz
        )

    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "dof_pos": dof_pos,
        "missing_joint_names": missing_joint_names,
        "unknown_motor_names": unknown_motor_names,
    }


def main() -> int:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    args = build_argparser().parse_args()

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    robot_xml_path = Path(args.robot_xml).expanduser()
    joint_order = load_joint_order(robot_xml_path)
    alias_to_joint_name = build_alias_to_joint_name(joint_order)

    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(args.ip)
    recv_flag = 1 if args.non_blocking else 0

    motion_frames = []
    received_total = 0
    skipped_total = 0
    last_wait_log = 0.0
    summary_printed = False
    start_time = time.time()

    try:
        print(f"Recording GMR-style motion to {output_path}")
        print(f"Loaded {len(joint_order)} joints from {robot_xml_path}")
        print("Press Ctrl+C to stop and save.")

        while g_running:
            frame = LuMoSDKClient.ReceiveData(recv_flag)
            if frame is None:
                current_time = time.time()
                if current_time - last_wait_log >= args.wait_log_interval:
                    print(f"Waiting for LuMo frame... captured={len(motion_frames)}")
                    last_wait_log = current_time
                if args.non_blocking:
                    time.sleep(max(args.poll_sleep, 0.0))
                continue

            received_total += 1
            skeleton = select_tracked_skeleton(frame, args.robot_name)
            if skeleton is None:
                skipped_total += 1
                current_time = time.time()
                if current_time - last_wait_log >= args.wait_log_interval:
                    if args.robot_name:
                        print(
                            f"No tracked skeleton matched RobotName={args.robot_name!r}; "
                            f"skipped={skipped_total}"
                        )
                    else:
                        print(f"No tracked skeleton available; skipped={skipped_total}")
                    last_wait_log = current_time
                continue

            motion_frame = build_motion_frame(skeleton, joint_order, alias_to_joint_name)
            motion_frames.append(motion_frame)

            if not summary_printed:
                if motion_frame["missing_joint_names"]:
                    print("Missing joints in current MotorAngle data:")
                    for joint_name in motion_frame["missing_joint_names"]:
                        print(joint_name)
                else:
                    print("All G1 joints were mapped from MotorAngle data.")

                if motion_frame["unknown_motor_names"]:
                    print("MotorAngle names that could not be mapped:")
                    for joint_name in motion_frame["unknown_motor_names"]:
                        print(joint_name)

                summary_printed = True

            if len(motion_frames) == 1 or len(motion_frames) % 100 == 0:
                print(
                    f"Captured {len(motion_frames)} frame(s) "
                    f"(latest FrameId={frame.FrameId}, dof={len(joint_order)})"
                )
    except KeyboardInterrupt:
        print("\nInterrupted by user, stopping capture...")
    finally:
        LuMoSDKClient.Close()

    if not motion_frames:
        print("No valid frames were captured; nothing was saved.")
        return 1

    motion_data = {
        "fps": args.motion_fps,
        "root_pos": np.stack([frame["root_pos"] for frame in motion_frames], axis=0),
        "root_rot": np.stack([frame["root_rot"] for frame in motion_frames], axis=0),
        "dof_pos": np.stack([frame["dof_pos"] for frame in motion_frames], axis=0),
        "local_body_pos": None,
        "link_body_list": None,
    }

    with output_path.open("wb") as f:
        pickle.dump(motion_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(motion_frames)} frame(s) to {output_path}")
    print(
        f"Received frames={received_total}, skipped frames={skipped_total}, "
        f"elapsed={time.time() - start_time:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
