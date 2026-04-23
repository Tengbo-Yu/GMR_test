#!/usr/bin/env python3
"""
Send live LuMo human mocap frames to another machine.

Default transport is UDP. The payload is a fixed-length float32 array with the
bone order defined in `general_motion_retargeting.utils.lumo_network`.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "LuMoSDKPy") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "LuMoSDKPy"))

from general_motion_retargeting.utils.lumo_network import (
    FRAME_BYTE_SIZE,
    LUMO_REQUIRED_SOURCE_BONES,
    build_bone_map,
    build_gmr_human_frame_from_lumo,
    human_frame_to_bytes,
    make_demo_human_frame,
    missing_stream_bones,
    select_tracked_skeleton,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send LuMo live human frames over the network.")
    parser.add_argument("--lumo_ip", type=str, default="192.168.50.150", help="LuMo SDK source IP.")
    parser.add_argument("--dst_ip", type=str, required=True, help="Destination IP address.")
    parser.add_argument("--dst_port", type=int, required=True, help="Destination port.")
    parser.add_argument("--protocol", choices=["udp", "tcp"], default="udp")
    parser.add_argument("--motion_fps", type=int, default=120)
    parser.add_argument("--non_blocking", action="store_true", default=False)
    parser.add_argument("--print_joint_summary", action="store_true", default=False)
    parser.add_argument("--wait_log_interval", type=float, default=2.0)
    parser.add_argument("--stats_interval", type=float, default=2.0)
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        default=False,
        help="Send synthetic frames instead of reading LuMoSDK. Useful for network/protocol testing.",
    )
    return parser


def make_socket(protocol: str, dst_ip: str, dst_port: int):
    if protocol == "udp":
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return sock

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((dst_ip, dst_port))
    return sock


def send_payload(sock, protocol: str, dst_ip: str, dst_port: int, payload: bytes) -> int:
    if protocol == "udp":
        return sock.sendto(payload, (dst_ip, dst_port))
    sock.sendall(payload)
    return len(payload)


def main() -> int:
    args = build_argparser().parse_args()

    sock = make_socket(args.protocol, args.dst_ip, args.dst_port)
    print(f"Sending to {args.dst_ip}:{args.dst_port} over {args.protocol.upper()}")
    print(f"Frame payload size: {FRAME_BYTE_SIZE} bytes")

    if args.demo_mode:
        print("Demo mode enabled: using synthetic human frames.")
        LuMoSDKClient = None
    else:
        import LuMoSDKClient

        print(f"Connecting LuMo SDK to {args.lumo_ip}")
        LuMoSDKClient.Init()
        LuMoSDKClient.Connnect(args.lumo_ip)

    recv_flag = 1 if args.non_blocking else 0
    sent_frames = 0
    failed_frames = 0
    dropped_frames = 0
    sent_bytes = 0
    summary_printed = False
    frame_index = 0
    waiting_since = time.time()
    last_wait_log = 0.0
    last_stats_log = time.time()

    try:
        while True:
            if args.demo_mode:
                human_frame = make_demo_human_frame(frame_index, fps=args.motion_fps)
                time.sleep(1.0 / max(args.motion_fps, 1))
            else:
                frame = LuMoSDKClient.ReceiveData(recv_flag)
                if frame is None:
                    dropped_frames += 1
                    current_time = time.time()
                    if current_time - last_wait_log >= args.wait_log_interval:
                        print(
                            f"Waiting for LuMo frame... "
                            f"{current_time - waiting_since:.1f}s elapsed | "
                            f"sent={sent_frames} dropped={dropped_frames}"
                        )
                        last_wait_log = current_time
                    if args.non_blocking:
                        time.sleep(0.001)
                    continue

                skeleton = select_tracked_skeleton(frame)
                if skeleton is None:
                    dropped_frames += 1
                    continue

                bones_by_name = build_bone_map(skeleton)
                if args.print_joint_summary and not summary_printed:
                    available = set(bones_by_name.keys())
                    missing_source = [name for name in LUMO_REQUIRED_SOURCE_BONES if name not in available]
                    print(f"Received {len(available)} LuMo bones.")
                    if missing_source:
                        print("Missing source bones:")
                        for bone_name in missing_source:
                            print(f"  {bone_name}")
                    else:
                        print("All source bones required for streaming are present.")
                    summary_printed = True

                human_frame = build_gmr_human_frame_from_lumo(bones_by_name)

            missing = missing_stream_bones(human_frame)
            if missing:
                failed_frames += 1
                print(f"Skipping frame with missing stream bones: {missing}")
                frame_index += 1
                continue

            payload = human_frame_to_bytes(human_frame)
            try:
                sent = send_payload(sock, args.protocol, args.dst_ip, args.dst_port, payload)
                if sent != len(payload):
                    failed_frames += 1
                    print(f"Partial send: {sent}/{len(payload)} bytes")
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

            frame_index += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if not args.demo_mode:
            LuMoSDKClient.Close()
        sock.close()
        print(
            f"Final send stats | ok={sent_frames} fail={failed_frames} "
            f"dropped={dropped_frames} bytes={sent_bytes}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
