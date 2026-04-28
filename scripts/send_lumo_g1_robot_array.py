#!/usr/bin/env python3
"""Send LuMo G1 motion as the original root + dof float32 array."""

from __future__ import annotations

import argparse
import re
import signal
import socket
import struct
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LUMO_SDK_DIR = REPO_ROOT / "LuMoSDKPy"
DEFAULT_ROBOT_XML = REPO_ROOT / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LUMO_SDK_DIR) not in sys.path:
    sys.path.insert(0, str(LUMO_SDK_DIR))

import LuMoSDKClient
from general_motion_retargeting import RobotMotionViewer


g_running = True


def signal_handler(signum, frame) -> None:
    del frame
    global g_running
    print(f"\nReceived signal {signum}, shutting down...")
    g_running = False


class JointAngleSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.prev_dof_pos: Optional[np.ndarray] = None

    def apply(self, dof_pos: np.ndarray) -> np.ndarray:
        if self.alpha >= 1.0:
            self.prev_dof_pos = dof_pos.copy()
            return dof_pos
        if self.prev_dof_pos is None:
            self.prev_dof_pos = dof_pos.copy()
            return dof_pos
        smoothed = self.alpha * dof_pos + (1.0 - self.alpha) * self.prev_dof_pos
        self.prev_dof_pos = smoothed
        return smoothed


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream LuMo G1 motion as root_pos + root_rot(xyzw) + dof_pos."
    )
    parser.add_argument("--ip", type=str, default="192.168.50.150", help="LuMo SDK source IP.")
    parser.add_argument("--robot-name", type=str, default="UnitreeG1", help="Tracked RobotName filter.")
    parser.add_argument("--dst-ip", type=str, default="192.168.50.34", help="Destination IP.")
    parser.add_argument("--dst-port", type=int, default=9000, help="Destination port.")
    parser.add_argument("--protocol", choices=["udp", "tcp"], default="udp", help="Transport protocol.")
    parser.add_argument(
        "--robot-xml",
        type=str,
        default=str(DEFAULT_ROBOT_XML),
        help="MuJoCo robot xml used to define dof order.",
    )
    parser.add_argument(
        "--non-blocking",
        action="store_true",
        default=False,
        help="Use non-blocking LuMo ReceiveData(1).",
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
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=2.0,
        help="How often to print send stats.",
    )
    parser.add_argument(
        "--print-joint-summary",
        action="store_true",
        default=False,
        help="Print one-time mapping summary after the first valid frame.",
    )
    parser.add_argument(
        "--joint-smoothing-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor for dof_pos. Smaller is smoother, 1.0 disables smoothing.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize the current G1 dof_pos on a local MuJoCo viewer.",
    )
    parser.add_argument(
        "--motion-fps",
        type=int,
        default=120,
        help="Viewer playback fps when visualization is enabled.",
    )
    parser.add_argument(
        "--rate-limit",
        action="store_true",
        default=False,
        help="Rate limit the local viewer to motion-fps.",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=False,
        help="Record the local visualization to a video file.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default="videos/lumo_g1_robot_online.mp4",
        help="Video path used when --record-video is enabled.",
    )
    parser.add_argument(
        "--viewer-use-stream-root",
        action="store_true",
        default=False,
        help="Use streamed root pose in the local viewer. By default the viewer keeps the XML root pose stable.",
    )
    parser.add_argument(
        "--viewer-follow-camera",
        action="store_true",
        default=False,
        help="Continuously recenter the local viewer camera on the robot.",
    )
    parser.add_argument(
        "--send-identity-root-rot",
        action="store_true",
        default=False,
        help="Force the transmitted root_rot_xyzw to [0, 0, 0, 1].",
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


def quat_inv_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quat_wxyz(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def transform_lumo_position(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, -z, y], dtype=np.float64) * 0.01


def transform_lumo_quat_to_xyzw(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    lumo_to_gmr_wxyz = np.array(
        [0.7071067811865476, 0.7071067811865476, 0.0, 0.0],
        dtype=np.float64,
    )
    quat_wxyz = normalize_quat_wxyz(np.array([qw, qx, qy, qz], dtype=np.float64))
    transformed_wxyz = normalize_quat_wxyz(
        quat_mul_wxyz(
            quat_mul_wxyz(lumo_to_gmr_wxyz, quat_wxyz),
            quat_inv_wxyz(lumo_to_gmr_wxyz),
        )
    )
    return transformed_wxyz[[1, 2, 3, 0]]


def quat_xyzw_to_euler_xyz_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    x, y, z, w = quat_xyzw / norm

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def quat_xyzw_to_angle_deg(quat_xyzw: np.ndarray) -> float:
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-8:
        return 0.0
    w = float(np.clip((quat_xyzw / norm)[3], -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(abs(w))))


def normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat_xyzw)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat_xyzw / norm


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_inv_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = normalize_quat_xyzw(quat_xyzw)
    return np.array(
        [-quat_xyzw[0], -quat_xyzw[1], -quat_xyzw[2], quat_xyzw[3]],
        dtype=np.float64,
    )


def root_rot_relative_to_initial(
    root_rot_xyzw: np.ndarray,
    initial_root_rot_xyzw: np.ndarray,
) -> np.ndarray:
    root_rot_xyzw = normalize_quat_xyzw(root_rot_xyzw)
    initial_root_rot_xyzw = normalize_quat_xyzw(initial_root_rot_xyzw)
    if np.dot(root_rot_xyzw, initial_root_rot_xyzw) < 0.0:
        root_rot_xyzw = -root_rot_xyzw
    return normalize_quat_xyzw(
        quat_mul_xyzw(quat_inv_xyzw(initial_root_rot_xyzw), root_rot_xyzw)
    )


def build_motion_array(
    skeleton,
    joint_order: List[str],
    alias_to_joint_name: Dict[str, str],
    joint_smoother: Optional[JointAngleSmoother] = None,
    send_identity_root_rot: bool = False,
) -> tuple[np.ndarray, List[str], List[str]]:
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

    if joint_smoother is not None:
        dof_pos = joint_smoother.apply(dof_pos)

    root_pos = np.zeros(3, dtype=np.float64)
    root_rot_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    root_bone = find_root_bone(skeleton)
    if root_bone is not None:
        root_pos = transform_lumo_position(root_bone.X, root_bone.Y, root_bone.Z)
        root_rot_xyzw = transform_lumo_quat_to_xyzw(
            root_bone.qw, root_bone.qx, root_bone.qy, root_bone.qz
        )
    if send_identity_root_rot:
        root_rot_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    payload = np.concatenate([root_pos, root_rot_xyzw, dof_pos]).astype(
        np.float32,
        copy=False,
    )
    return payload, missing_joint_names, unknown_motor_names


def make_socket(protocol: str, dst_ip: str, dst_port: int):
    if protocol == "udp":
        return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((dst_ip, dst_port))
    return sock


def send_payload(sock, protocol: str, dst_ip: str, dst_port: int, payload: np.ndarray) -> int:
    data = payload.tobytes()
    if protocol == "udp":
        return sock.sendto(data, (dst_ip, dst_port))
    sock.sendall(struct.pack("!I", len(data)) + data)
    return len(data)


def main() -> int:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    args = build_argparser().parse_args()

    robot_xml_path = Path(args.robot_xml).expanduser()
    joint_order = load_joint_order(robot_xml_path)
    alias_to_joint_name = build_alias_to_joint_name(joint_order)
    if not 0.0 < args.joint_smoothing_alpha <= 1.0:
        raise ValueError("--joint-smoothing-alpha must be in (0, 1].")
    joint_smoother = JointAngleSmoother(args.joint_smoothing_alpha)
    viewer = None
    if args.visualize:
        viewer = RobotMotionViewer(
            robot_type="unitree_g1",
            motion_fps=args.motion_fps,
            camera_follow=True,
            record_video=args.record_video,
            video_path=args.video_path,
        )

    sock = make_socket(args.protocol, args.dst_ip, args.dst_port)
    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(args.ip)
    recv_flag = 1 if args.non_blocking else 0

    sent_frames = 0
    failed_frames = 0
    dropped_frames = 0
    sent_bytes = 0
    last_wait_log = 0.0
    last_stats_log = time.time()
    summary_printed = False
    initial_rotation_printed = False
    initial_root_rot_xyzw: Optional[np.ndarray] = None
    if viewer is not None:
        identity_root_pos = viewer.data.qpos[:3].copy()
        identity_root_rot = viewer.data.qpos[3:7].copy()
    else:
        identity_root_pos = np.zeros(3, dtype=np.float64)
        identity_root_rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    last_valid_dof_pos: Optional[np.ndarray] = None

    try:
        print(f"Streaming to {args.dst_ip}:{args.dst_port} over {args.protocol.upper()}")
        print(f"Loaded {len(joint_order)} joints from {robot_xml_path}")
        print("Array layout: root_pos(3) + root_rot_xyzw(4) + dof_pos(29)")
        print(f"Joint smoothing alpha: {args.joint_smoothing_alpha}")
        if args.send_identity_root_rot:
            print("Transmit root_rot_xyzw forced to [0, 0, 0, 1].")
        else:
            print("Transmit root_rot_xyzw relative to the first valid frame.")
        if viewer is not None:
            print("Local G1 visualization enabled.")

        while g_running:
            frame = LuMoSDKClient.ReceiveData(recv_flag)
            if frame is None:
                current_time = time.time()
                if current_time - last_wait_log >= args.wait_log_interval:
                    print(
                        f"Waiting for LuMo frame... sent={sent_frames} dropped={dropped_frames}"
                    )
                    last_wait_log = current_time
                if viewer is not None and last_valid_dof_pos is not None:
                    viewer.step(
                        root_pos=identity_root_pos,
                        root_rot=identity_root_rot,
                        dof_pos=last_valid_dof_pos,
                        rate_limit=args.rate_limit,
                        follow_camera=args.viewer_follow_camera,
                    )
                if args.non_blocking:
                    time.sleep(max(args.poll_sleep, 0.0))
                continue

            skeleton = select_tracked_skeleton(frame, args.robot_name)
            if skeleton is None:
                dropped_frames += 1
                continue

            payload, missing_joint_names, unknown_motor_names = build_motion_array(
                skeleton=skeleton,
                joint_order=joint_order,
                alias_to_joint_name=alias_to_joint_name,
                joint_smoother=joint_smoother,
                send_identity_root_rot=args.send_identity_root_rot,
            )

            if args.print_joint_summary and not summary_printed:
                if missing_joint_names:
                    print("Missing joints in current MotorAngle data:")
                    for joint_name in missing_joint_names:
                        print(f"  {joint_name}")
                else:
                    print("All G1 joints were mapped from MotorAngle data.")

                if unknown_motor_names:
                    print("MotorAngle names that could not be mapped:")
                    for joint_name in unknown_motor_names:
                        print(f"  {joint_name}")

                print(f"Outgoing array length: {payload.shape[0]} float32 values")
                summary_printed = True

            payload64 = payload.astype(np.float64, copy=True)
            raw_root_rot_xyzw = payload64[3:7].copy()
            if not args.send_identity_root_rot:
                if initial_root_rot_xyzw is None:
                    initial_root_rot_xyzw = normalize_quat_xyzw(raw_root_rot_xyzw)
                relative_root_rot_xyzw = root_rot_relative_to_initial(
                    raw_root_rot_xyzw,
                    initial_root_rot_xyzw,
                )
                payload[3:7] = relative_root_rot_xyzw.astype(np.float32)
                payload64[3:7] = relative_root_rot_xyzw

            current_root_pos = payload64[:3]
            current_root_rot = payload64[3:7][[3, 0, 1, 2]]
            current_dof_pos = payload64[7:]
            last_valid_dof_pos = current_dof_pos

            if not initial_rotation_printed:
                transmitted_root_rot_xyzw = payload64[3:7]
                raw_euler_xyz_deg = quat_xyzw_to_euler_xyz_deg(raw_root_rot_xyzw)
                transmitted_euler_xyz_deg = quat_xyzw_to_euler_xyz_deg(
                    transmitted_root_rot_xyzw
                )
                transmitted_angle_deg = quat_xyzw_to_angle_deg(transmitted_root_rot_xyzw)
                print("Initial streamed root rotation (first valid frame):")
                print(
                    "  raw_root_rot_xyzw = "
                    f"[{raw_root_rot_xyzw[0]:.6f}, {raw_root_rot_xyzw[1]:.6f}, "
                    f"{raw_root_rot_xyzw[2]:.6f}, {raw_root_rot_xyzw[3]:.6f}]"
                )
                print(
                    "  raw_euler_xyz_deg(roll, pitch, yaw) = "
                    f"[{raw_euler_xyz_deg[0]:.3f}, {raw_euler_xyz_deg[1]:.3f}, "
                    f"{raw_euler_xyz_deg[2]:.3f}]"
                )
                print(
                    "  transmitted_root_rot_xyzw = "
                    f"[{transmitted_root_rot_xyzw[0]:.6f}, "
                    f"{transmitted_root_rot_xyzw[1]:.6f}, "
                    f"{transmitted_root_rot_xyzw[2]:.6f}, "
                    f"{transmitted_root_rot_xyzw[3]:.6f}]"
                )
                print(
                    "  transmitted_euler_xyz_deg(roll, pitch, yaw) = "
                    f"[{transmitted_euler_xyz_deg[0]:.3f}, "
                    f"{transmitted_euler_xyz_deg[1]:.3f}, "
                    f"{transmitted_euler_xyz_deg[2]:.3f}]"
                )
                print(f"  transmitted_rotation_angle_deg = {transmitted_angle_deg:.3f}")
                initial_rotation_printed = True

            if viewer is not None:
                viewer_root_pos = current_root_pos if args.viewer_use_stream_root else identity_root_pos
                viewer_root_rot = current_root_rot if args.viewer_use_stream_root else identity_root_rot
                viewer.step(
                    root_pos=viewer_root_pos,
                    root_rot=viewer_root_rot,
                    dof_pos=current_dof_pos,
                    rate_limit=args.rate_limit,
                    follow_camera=args.viewer_follow_camera,
                )

            try:
                sent = send_payload(sock, args.protocol, args.dst_ip, args.dst_port, payload)
                if sent != payload.nbytes:
                    failed_frames += 1
                    print(f"Partial send: {sent}/{payload.nbytes} bytes")
                else:
                    sent_frames += 1
                    sent_bytes += sent
            except OSError as exc:
                failed_frames += 1
                print(f"Send failed: {exc}")
                if args.protocol == "tcp":
                    break

            current_time = time.time()
            if current_time - last_stats_log >= args.stats_interval:
                print(
                    f"Send stats | ok={sent_frames} fail={failed_frames} "
                    f"dropped={dropped_frames} bytes={sent_bytes}"
                )
                last_stats_log = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        LuMoSDKClient.Close()
        sock.close()
        if viewer is not None:
            viewer.close()
        print(
            f"Final send stats | ok={sent_frames} fail={failed_frames} "
            f"dropped={dropped_frames} bytes={sent_bytes}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
