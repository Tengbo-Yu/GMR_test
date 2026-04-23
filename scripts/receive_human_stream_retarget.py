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
            damping=1.0,
            use_velocity_limit=True,
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
                    qpos = retargeter.retarget(human_frame, offset_to_ground=False)
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

            current_time = time.time()
            if current_time - last_stats_log >= args.stats_interval:
                print(
                    f"Receive stats | ok={received_frames} fail={failed_frames} "
                    f"bytes={received_bytes}"
                )
                last_stats_log = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if viewer is not None:
            viewer.close()

        if conn is not None:
            conn.close()
        server_sock.close()

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

        print(f"Final receive stats | ok={received_frames} fail={failed_frames} bytes={received_bytes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
