#!/usr/bin/env python3
"""
LuMo live skeleton streaming to GMR online retargeting.

This script bridges LuMoSDK live skeleton frames into the same human-frame
structure used by `bvh_to_robot.py --format mocap`, and can optionally record
the incoming stream as a growing BVH file using a reference hierarchy.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "LuMoSDKPy") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "LuMoSDKPy"))

import LuMoSDKClient
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils


g_running = True


def signal_handler(signum, frame):
    del frame
    global g_running
    print(f"\nReceived signal {signum}, shutting down...")
    g_running = False


@dataclass
class BVHTemplate:
    header: str
    joint_names: List[str]
    parents: List[int]
    frame_time: float
    root_name: str


class LiveBVHRecorder:
    def __init__(self, template: BVHTemplate, output_path: Optional[str], flush_every: int = 0):
        self.template = template
        self.output_path = output_path
        self.flush_every = flush_every
        self.frames: List[str] = []

    def add_frame(self, bones_by_name: Dict[str, object]) -> None:
        global_pos = []
        global_quat = []

        for joint_name in self.template.joint_names:
            bone = bones_by_name.get(joint_name)
            if bone is None:
                if global_pos:
                    global_pos.append(global_pos[-1].copy())
                    global_quat.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
                else:
                    global_pos.append(np.zeros(3, dtype=np.float64))
                    global_quat.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
                continue

            global_pos.append(np.array([bone.X, bone.Y, bone.Z], dtype=np.float64))
            global_quat.append(
                normalize_quat(
                    np.array([bone.qw, bone.qx, bone.qy, bone.qz], dtype=np.float64)
                )
            )

        global_pos_np = np.asarray(global_pos, dtype=np.float64)[np.newaxis, ...]
        global_quat_np = np.asarray(global_quat, dtype=np.float64)[np.newaxis, ...]

        local_quat_np, _ = lafan_utils.quat_ik(global_quat_np, global_pos_np, self.template.parents)
        local_quat_np = local_quat_np[0]

        root_index = 0
        root_pos = global_pos[root_index]
        motion_values = [f"{root_pos[0]:.6f}", f"{root_pos[1]:.6f}", f"{root_pos[2]:.6f}"]

        for local_quat in local_quat_np:
            euler_zyx = quat_wxyz_to_euler_zyx_deg(local_quat)
            motion_values.extend(
                [f"{euler_zyx[0]:.6f}", f"{euler_zyx[1]:.6f}", f"{euler_zyx[2]:.6f}"]
            )

        self.frames.append(" ".join(motion_values))

        if self.output_path and self.flush_every > 0 and len(self.frames) % self.flush_every == 0:
            self.write_file()

    def write_file(self) -> None:
        if not self.output_path:
            return

        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write(self.template.header.rstrip() + "\n")
            f.write("MOTION\n")
            f.write(f"Frames: {len(self.frames)}\n")
            f.write(f"Frame Time: {self.template.frame_time:.8f}\n")
            if self.frames:
                f.write("\n".join(self.frames))
                f.write("\n")


class MotionForwarder:
    def __init__(
        self,
        host: str,
        port: int,
        protocol: str = "udp",
        payload_format: str = "pickle",
        tcp_server: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.protocol = protocol
        self.payload_format = payload_format
        self.tcp_server = tcp_server
        self.sock = None
        self.client_sock = None

        if self.protocol == "udp":
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        elif self.protocol == "tcp":
            if self.tcp_server:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.sock.bind((self.host, self.port))
                self.sock.listen(1)
                self.sock.settimeout(0.0)
                print(f"Forward TCP server listening on {self.host}:{self.port}")
            else:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                print(f"Forward TCP client connected to {self.host}:{self.port}")
        else:
            raise ValueError(f"Unsupported protocol: {self.protocol}")

    def _serialize(self, payload: Dict[str, object]) -> bytes:
        if self.payload_format == "pickle":
            import pickle

            return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        if self.payload_format == "json":
            return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        raise ValueError(f"Unsupported payload format: {self.payload_format}")

    def _maybe_accept_client(self) -> None:
        if not self.tcp_server or self.client_sock is not None:
            return
        try:
            client_sock, client_addr = self.sock.accept()
        except BlockingIOError:
            return
        client_sock.setblocking(True)
        self.client_sock = client_sock
        print(f"Forward TCP client connected from {client_addr[0]}:{client_addr[1]}")

    def send(self, payload: Dict[str, object]) -> bool:
        data = self._serialize(payload)

        if self.protocol == "udp":
            self.sock.sendto(data, (self.host, self.port))
            return True

        if self.tcp_server:
            self._maybe_accept_client()
            if self.client_sock is None:
                return False
            try:
                if self.payload_format == "pickle":
                    self.client_sock.sendall(struct.pack("!I", len(data)) + data)
                else:
                    self.client_sock.sendall(data)
                return True
            except OSError:
                self.client_sock.close()
                self.client_sock = None
                return False

        if self.payload_format == "pickle":
            self.sock.sendall(struct.pack("!I", len(data)) + data)
        else:
            self.sock.sendall(data)
        return True

    def close(self) -> None:
        if self.client_sock is not None:
            self.client_sock.close()
            self.client_sock = None
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def normalize_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat_wxyz)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat_wxyz / norm


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)


def quat_wxyz_to_euler_zyx_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = R.from_quat(quat_wxyz_to_xyzw(normalize_quat(quat_wxyz)))
    return rotation.as_euler("ZYX", degrees=True)


def load_bvh_template(template_path: str, fallback_frame_time: float) -> BVHTemplate:
    with open(template_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    motion_index = None
    for i, line in enumerate(lines):
        if line.strip() == "MOTION":
            motion_index = i
            break

    if motion_index is None:
        raise ValueError(f"Template BVH has no MOTION section: {template_path}")

    header = "".join(lines[:motion_index])
    frame_time = fallback_frame_time
    for line in lines[motion_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("Frame Time:"):
            frame_time = float(stripped.split(":", 1)[1].strip())
            break

    joint_names: List[str] = []
    parents: List[int] = []
    scope_stack: List[Optional[int]] = []
    root_name = ""

    for raw_line in lines[:motion_index]:
        line = raw_line.strip()
        if line.startswith("ROOT "):
            joint_name = line.split(None, 1)[1]
            root_name = joint_name
            joint_names.append(joint_name)
            parents.append(-1)
            scope_stack.append(len(joint_names) - 1)
        elif line.startswith("JOINT "):
            joint_name = line.split(None, 1)[1]
            parent_index = next((idx for idx in reversed(scope_stack) if idx is not None), -1)
            joint_names.append(joint_name)
            parents.append(parent_index)
            scope_stack.append(len(joint_names) - 1)
        elif line.startswith("End Site"):
            scope_stack.append(None)
        elif line == "}":
            if scope_stack:
                scope_stack.pop()

    if not joint_names:
        raise ValueError(f"Failed to parse joints from BVH template: {template_path}")

    return BVHTemplate(
        header=header,
        joint_names=joint_names,
        parents=parents,
        frame_time=frame_time,
        root_name=root_name,
    )


def select_tracked_skeleton(frame) -> Optional[object]:
    for skeleton in frame.skeletons:
        if skeleton.IsTrack:
            return skeleton
    return None


def build_bone_map(skeleton) -> Dict[str, object]:
    return {bone.Name: bone for bone in skeleton.skeletonBones}


def build_gmr_human_frame(
    bones_by_name: Dict[str, object],
    rotation_matrix: np.ndarray,
    rotation_quat_wxyz: np.ndarray,
    position_scale: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    result: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for bone_name, bone in bones_by_name.items():
        pos = np.array([bone.X, bone.Y, bone.Z], dtype=np.float64)
        quat_wxyz = normalize_quat(
            np.array([bone.qw, bone.qx, bone.qy, bone.qz], dtype=np.float64)
        )

        transformed_pos = pos @ rotation_matrix.T * position_scale
        transformed_quat = normalize_quat(
            lafan_utils.quat_mul(rotation_quat_wxyz[np.newaxis, :], quat_wxyz[np.newaxis, :])[0]
        )
        result[bone_name] = [transformed_pos, transformed_quat]

    if "LeftFoot" in result and "LeftToe" in result:
        result["LeftFootMod"] = [result["LeftFoot"][0], result["LeftToe"][1]]
    if "RightFoot" in result and "RightToe" in result:
        result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]

    return result


def maybe_print_joint_summary(bones_by_name: Dict[str, object], required_bones: List[str]) -> None:
    available = set(bones_by_name.keys())
    missing = [name for name in required_bones if name not in available]
    print(f"Received {len(available)} skeleton bones.")
    if missing:
        print("Missing GMR-required bones:")
        for name in missing:
            print(f"  {name}")
    else:
        print("All GMR-required bones are present.")


def make_motion_frame_payload(qpos: np.ndarray, fps: int, frame_index: int) -> Dict[str, object]:
    return {
        "fps": fps,
        "frame_index": frame_index,
        "root_pos": qpos[:3].tolist(),
        "root_rot": qpos[3:7][[1, 2, 3, 0]].tolist(),
        "dof_pos": qpos[7:].tolist(),
        "local_body_pos": None,
        "link_body_list": None,
    }


def main() -> int:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="LuMo live streaming to GMR online retargeting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ip", type=str, default="192.168.50.150", help="LuMo server IP.")
    parser.add_argument("--robot", choices=["unitree_g1"], default="unitree_g1")
    parser.add_argument(
        "--template_bvh",
        type=str,
        default=str(REPO_ROOT / "test_Skeleton0.bvh"),
        help="Reference BVH used for hierarchy and frame time when saving live BVH.",
    )
    parser.add_argument(
        "--save_bvh",
        type=str,
        default=None,
        help="Optional path to save a live-recorded BVH stream.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=0,
        help="Rewrite the BVH file every N captured frames. 0 disables live flushing.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional path to save retargeted robot motion as pickle.",
    )
    parser.add_argument("--motion_fps", type=int, default=120)
    parser.add_argument(
        "--human_height",
        type=float,
        default=1.75,
        help="Human height in meters for GMR scaling. Matches offline mocap default by default.",
    )
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/lumo_live.mp4")
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        default=False,
        help="Run retargeting without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--non_blocking",
        action="store_true",
        default=False,
        help="Use non-blocking LuMo ReceiveData(1) instead of blocking ReceiveData(0).",
    )
    parser.add_argument(
        "--print_joint_summary",
        action="store_true",
        default=False,
        help="Print one-time available/missing joint summary after the first tracked skeleton frame.",
    )
    parser.add_argument(
        "--forward_host",
        type=str,
        default=None,
        help="Optional destination host for real-time robot motion forwarding.",
    )
    parser.add_argument(
        "--forward_port",
        type=int,
        default=None,
        help="Optional destination or listening port for real-time robot motion forwarding.",
    )
    parser.add_argument(
        "--forward_protocol",
        choices=["udp", "tcp"],
        default="udp",
        help="Protocol used for real-time forwarding.",
    )
    parser.add_argument(
        "--forward_format",
        choices=["pickle", "json"],
        default="pickle",
        help="Serialization format for forwarded motion frames.",
    )
    parser.add_argument(
        "--forward_bind",
        action="store_true",
        default=False,
        help="For TCP only: bind and wait for a consumer to connect instead of connecting out.",
    )
    args = parser.parse_args()

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    template = load_bvh_template(args.template_bvh, fallback_frame_time=1.0 / args.motion_fps)
    recorder = LiveBVHRecorder(template=template, output_path=args.save_bvh, flush_every=args.flush_every)
    forwarder = None

    if (args.forward_host is None) != (args.forward_port is None):
        raise ValueError("Use --forward_host and --forward_port together, or omit both.")

    if args.forward_port is not None:
        forwarder = MotionForwarder(
            host=args.forward_host,
            port=args.forward_port,
            protocol=args.forward_protocol,
            payload_format=args.forward_format,
            tcp_server=args.forward_bind,
        )

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    rotation_quat_wxyz = quat_xyzw_to_wxyz(R.from_matrix(rotation_matrix).as_quat())
    position_scale = 0.01

    required_bones = [
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

    print("[1/4] Initializing LuMo SDK...")
    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(args.ip)

    print("[2/4] Initializing GMR retargeter...")
    retargeter = GMR(
        src_human="bvh_mocap",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
        solver="daqp",
        damping=1.0,
        use_velocity_limit=True,
    )

    viewer = None
    if not args.no_viewer:
        print("[3/4] Initializing viewer...")
        viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=args.motion_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=args.video_path,
        )
    else:
        print("[3/4] Viewer disabled.")

    print("[4/4] Starting LuMo live retargeting...")
    print(f"LuMo server IP: {args.ip}")
    print(f"mocap_frame_rate: {args.motion_fps}")
    if forwarder is not None:
        if args.forward_protocol == "tcp" and args.forward_bind:
            print(
                f"Forwarding enabled: TCP server on {args.forward_host}:{args.forward_port} "
                f"({args.forward_format})"
            )
        else:
            print(
                f"Forwarding enabled: {args.forward_protocol.upper()} -> "
                f"{args.forward_host}:{args.forward_port} ({args.forward_format})"
            )
    print("Press Ctrl+C to stop.\n")

    total_frames = 0
    dropped_frames = 0
    qpos_list = []
    last_valid_qpos = None
    last_valid_human_frame = None
    summary_printed = False
    recv_flag = 1 if args.non_blocking else 0

    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    try:
        while g_running:
            frame = LuMoSDKClient.ReceiveData(recv_flag)
            if frame is None:
                dropped_frames += 1
                if viewer is not None and last_valid_qpos is not None:
                    viewer.step(
                        root_pos=last_valid_qpos[:3],
                        root_rot=last_valid_qpos[3:7],
                        dof_pos=last_valid_qpos[7:],
                        human_motion_data=last_valid_human_frame,
                        rate_limit=args.rate_limit,
                    )
                if args.non_blocking:
                    time.sleep(0.001)
                continue

            skeleton = select_tracked_skeleton(frame)
            if skeleton is None:
                dropped_frames += 1
                continue

            bones_by_name = build_bone_map(skeleton)
            if args.print_joint_summary and not summary_printed:
                maybe_print_joint_summary(bones_by_name, required_bones)
                summary_printed = True

            human_frame = build_gmr_human_frame(
                bones_by_name=bones_by_name,
                rotation_matrix=rotation_matrix,
                rotation_quat_wxyz=rotation_quat_wxyz,
                position_scale=position_scale,
            )

            missing_required = [name for name in required_bones if name not in human_frame]
            if missing_required:
                dropped_frames += 1
                if total_frames == 0:
                    print("First tracked frame is missing required bones, skipping frame.")
                    for name in missing_required:
                        print(f"  missing: {name}")
                continue

            recorder.add_frame(bones_by_name)

            try:
                qpos = retargeter.retarget(human_frame, offset_to_ground=False)
            except Exception as e:
                dropped_frames += 1
                print(f"Retargeting failed: {e}")
                continue

            total_frames += 1
            fps_counter += 1
            last_valid_qpos = qpos.copy()
            last_valid_human_frame = retargeter.scaled_human_data

            if viewer is not None:
                viewer.step(
                    root_pos=qpos[:3],
                    root_rot=qpos[3:7],
                    dof_pos=qpos[7:],
                    human_motion_data=retargeter.scaled_human_data,
                    rate_limit=args.rate_limit,
                    follow_camera=True,
                )

            if args.save_path is not None:
                qpos_list.append(qpos.copy())

            if forwarder is not None:
                payload = make_motion_frame_payload(
                    qpos=qpos,
                    fps=args.motion_fps,
                    frame_index=total_frames - 1,
                )
                forwarder.send(payload)

            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(
                    f"FPS: {actual_fps:.1f} | "
                    f"Retargeted: {total_frames} | "
                    f"Dropped: {dropped_frames} | "
                    f"BVH Frames: {len(recorder.frames)}"
                )
                fps_counter = 0
                fps_start_time = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        LuMoSDKClient.Close()
        if forwarder is not None:
            forwarder.close()

        if args.save_bvh is not None:
            recorder.write_file()
            print(f"Saved live BVH to {args.save_bvh}")

        if args.save_path is not None and qpos_list:
            import pickle

            root_pos = np.array([qpos[:3] for qpos in qpos_list])
            root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
            dof_pos = np.array([qpos[7:] for qpos in qpos_list])
            motion_data = {
                "fps": args.motion_fps,
                "root_pos": root_pos,
                "root_rot": root_rot,
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
            }
            with open(args.save_path, "wb") as f:
                pickle.dump(motion_data, f)
            print(f"Saved retargeted robot motion to {args.save_path}")

        print(f"\nTotal retargeted: {total_frames}")
        print(f"Total dropped: {dropped_frames}")

        if viewer is not None:
            viewer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
