#!/usr/bin/env python3
"""
Receive streamed human mocap frames and run online retargeting.

This script is intended to run on a second machine where the repository is
cloned. It listens for human frames produced by `send_lumo_human_stream.py`,
reconstructs the `bvh_mocap` human-frame dictionary, and feeds it to GMR.
"""

from __future__ import annotations

import argparse
import os
import pickle
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.utils.lumo_network import (
    FRAME_BYTE_SIZE,
    bytes_to_human_frame,
    recv_exact,
)


g_running = True


def signal_handler(signum, frame):
    del frame
    global g_running
    print(f"\nReceived signal {signum}, shutting down...")
    g_running = False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Receive human stream and run online retargeting.")
    parser.add_argument("--listen_host", type=str, default="0.0.0.0")
    parser.add_argument("--listen_port", type=int, required=True)
    parser.add_argument("--protocol", choices=["udp", "tcp"], default="udp")
    parser.add_argument("--robot", choices=["unitree_g1"], default="unitree_g1")
    parser.add_argument("--human_height", type=float, default=1.75)
    parser.add_argument("--motion_fps", type=int, default=120)
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/received_live.mp4")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--no_viewer", action="store_true", default=False)
    parser.add_argument("--stats_interval", type=float, default=2.0)
    parser.add_argument("--forward_udp_host", type=str, default="127.0.0.1")
    parser.add_argument("--forward_udp_port", type=int, default=9000)
    parser.add_argument("--disable_forward_udp", action="store_true", default=False)
    parser.add_argument(
        "--no_offset_to_ground",
        action="store_true",
        default=False,
        help="Disable moving live human targets so the lowest foot is above the ground.",
    )
    parser.add_argument(
        "--robot_ground_clearance",
        type=float,
        default=0.04,
        help="Minimum robot geometry height above ground after retargeting. Use a negative value to disable.",
    )
    parser.add_argument(
        "--print_waist_debug",
        action="store_true",
        default=False,
        help="Print waist joint qpos values at stats_interval to verify whether the waist is moving.",
    )
    return parser


def setup_socket(protocol: str, listen_host: str, listen_port: int):
    if protocol == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((listen_host, listen_port))
        return sock, None

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((listen_host, listen_port))
    server_sock.listen(1)
    print(f"Waiting for TCP sender on {listen_host}:{listen_port} ...")
    conn, addr = server_sock.accept()
    print(f"TCP sender connected from {addr[0]}:{addr[1]}")
    return server_sock, conn


def lift_robot_qpos_to_ground(retargeter, qpos: np.ndarray, clearance: float) -> np.ndarray:
    if clearance < 0:
        return qpos

    qpos = qpos.copy()
    data = retargeter.configuration.data
    data.qpos[:] = qpos

    import mujoco as mj

    mj.mj_forward(retargeter.model, data)
    if data.geom_xpos.shape[0] == 0:
        return qpos

    geom_vertical_radius = np.max(retargeter.model.geom_size, axis=1)
    lowest_z = float(np.min(data.geom_xpos[:, 2] - geom_vertical_radius))
    for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        body_id = retargeter.model.body(body_name).id
        lowest_z = min(lowest_z, float(data.xpos[body_id, 2] - 0.04))
    if lowest_z < clearance:
        qpos[2] += clearance - lowest_z
        data.qpos[:] = qpos
        mj.mj_forward(retargeter.model, data)
    return qpos


def get_named_dof_values(retargeter, qpos: np.ndarray, name_prefix: str) -> dict:
    values = {}
    for joint_name, dof_index in retargeter.robot_dof_names.items():
        if joint_name.startswith(name_prefix):
            qpos_index = dof_index + 6 if dof_index >= 6 else dof_index
            if qpos_index < qpos.shape[0]:
                values[joint_name] = float(qpos[qpos_index])
    return values


def main() -> int:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    args = build_argparser().parse_args()

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    server_sock, conn = setup_socket(args.protocol, args.listen_host, args.listen_port)
    print(f"Listening on {args.listen_host}:{args.listen_port} over {args.protocol.upper()}")
    print(f"Expected frame payload size: {FRAME_BYTE_SIZE} bytes")

    forward_sock = None
    if not args.disable_forward_udp:
        forward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Forwarding retargeted data over UDP to {args.forward_udp_host}:{args.forward_udp_port}")
        print("Forward UDP layout: root_pos(3) + root_rot_wxyz(4) + dof_pos(29) = 36 float32")
    else:
        print("UDP forwarding disabled.")

    retargeter = None
    viewer = None
    if not args.dry_run:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR
        from general_motion_retargeting import RobotMotionViewer

        print("Initializing GMR retargeter...")
        retargeter = GMR(
            src_human="bvh_mocap",
            tgt_robot=args.robot,
            actual_human_height=args.human_height,
            solver="daqp",
            damping=0.5,
            use_velocity_limit=False,
        )

        if not args.no_viewer:
            print("Initializing viewer...")
            viewer = RobotMotionViewer(
                robot_type=args.robot,
                motion_fps=args.motion_fps,
                transparent_robot=0,
                record_video=args.record_video,
                video_path=args.video_path,
            )
        else:
            print("Viewer disabled.")
    else:
        print("Dry-run mode enabled: frames will be decoded but not retargeted.")

    received_frames = 0
    failed_frames = 0
    received_bytes = 0
    forwarded_frames = 0
    forward_failed = 0
    qpos_list = []
    last_valid_qpos = None
    last_valid_human_frame = None
    last_stats_log = time.time()

    try:
        while g_running:
            try:
                if args.protocol == "udp":
                    payload, addr = server_sock.recvfrom(FRAME_BYTE_SIZE + 4096)
                    if len(payload) != FRAME_BYTE_SIZE:
                        failed_frames += 1
                        print(f"Unexpected UDP payload size from {addr[0]}:{addr[1]}: {len(payload)}")
                        continue
                else:
                    payload = recv_exact(conn, FRAME_BYTE_SIZE)
            except (OSError, ConnectionError) as exc:
                failed_frames += 1
                print(f"Receive failed: {exc}")
                if args.protocol == "tcp":
                    break
                continue

            try:
                human_frame = bytes_to_human_frame(payload)
            except ValueError as exc:
                failed_frames += 1
                print(f"Decode failed: {exc}")
                continue

            received_frames += 1
            received_bytes += len(payload)

            if received_frames == 1:
                print("Received first human frame successfully.")

            if retargeter is not None:
                try:
                    qpos = retargeter.retarget(
                        human_frame,
                        offset_to_ground=not args.no_offset_to_ground,
                    )
                    qpos = lift_robot_qpos_to_ground(
                        retargeter,
                        qpos,
                        clearance=args.robot_ground_clearance,
                    )
                except Exception as exc:
                    failed_frames += 1
                    print(f"Retarget failed: {exc}")
                    continue

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

                if forward_sock is not None:
                    try:
                        forward_payload = qpos.astype(np.float32, copy=False)
                        forward_data = forward_payload.tobytes()
                        sent_bytes = forward_sock.sendto(
                            forward_data,
                            (args.forward_udp_host, args.forward_udp_port),
                        )
                        if sent_bytes != len(forward_data):
                            forward_failed += 1
                            print(f"Forward partial send: {sent_bytes}/{len(forward_data)}")
                        else:
                            forwarded_frames += 1
                    except OSError as exc:
                        forward_failed += 1
                        print(f"Forward failed: {exc}")

            current_time = time.time()
            if current_time - last_stats_log >= args.stats_interval:
                print(
                    f"Receive stats | ok={received_frames} fail={failed_frames} "
                    f"bytes={received_bytes} fwd_ok={forwarded_frames} fwd_fail={forward_failed}"
                )
                if args.print_waist_debug and last_valid_qpos is not None and retargeter is not None:
                    waist_values = get_named_dof_values(retargeter, last_valid_qpos, "waist_")
                    print(f"Waist qpos: {waist_values}")
                last_stats_log = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if viewer is not None:
            viewer.close()

        if conn is not None:
            conn.close()
        server_sock.close()
        if forward_sock is not None:
            forward_sock.close()

        if args.save_path is not None and qpos_list:
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

        print(
            f"Final receive stats | ok={received_frames} fail={failed_frames} bytes={received_bytes} "
            f"fwd_ok={forwarded_frames} fwd_fail={forward_failed}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
