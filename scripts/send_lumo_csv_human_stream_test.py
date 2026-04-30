#!/usr/bin/env python3
"""
Replay a LuMo exported CSV as the same human-frame stream sent by local.sh.

The receiver side is `scripts/receive_human_stream_retarget.py`. Payloads use
the fixed-size format from `general_motion_retargeting.utils.lumo_network`.
"""

from __future__ import annotations

import argparse
import csv
import socket
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "general_motion_retargeting" not in sys.modules:
    package = types.ModuleType("general_motion_retargeting")
    package.__path__ = [str(REPO_ROOT / "general_motion_retargeting")]
    sys.modules["general_motion_retargeting"] = package

if "general_motion_retargeting.utils" not in sys.modules:
    utils_package = types.ModuleType("general_motion_retargeting.utils")
    utils_package.__path__ = [str(REPO_ROOT / "general_motion_retargeting" / "utils")]
    sys.modules["general_motion_retargeting.utils"] = utils_package

from general_motion_retargeting.utils.lumo_network_test import (
    WHOLE_BODY_BYTE_SIZE,
    LUMO_REQUIRED_SOURCE_BONES,
    build_gmr_human_frame_from_lumo,
    missing_whole_body_bones,
    whole_body_frame_to_bytes,
)


@dataclass
class CsvBone:
    Name: str
    X: float
    Y: float
    Z: float
    qw: float
    qx: float
    qy: float
    qz: float


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay LuMo CSV human frames over the network.")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=REPO_ROOT / "misc" / "0428.csv",
        help="LuMo exported CSV path.",
    )
    parser.add_argument("--dst_ip", type=str, required=True, help="Destination IP address.")
    parser.add_argument("--dst_port", type=int, required=True, help="Destination port.")
    parser.add_argument("--protocol", choices=["udp", "tcp"], default="udp")
    parser.add_argument(
        "--motion_fps",
        type=float,
        default=None,
        help="Replay FPS. Defaults to the CSV FrameRate metadata when present.",
    )
    parser.add_argument("--start_frame", type=int, default=1, help="1-based first CSV frame to send.")
    parser.add_argument("--end_frame", type=int, default=None, help="1-based last CSV frame to send.")
    parser.add_argument("--loop", action="store_true", default=False, help="Loop replay until interrupted.")
    parser.add_argument(
        "--no_rate_limit",
        action="store_true",
        default=False,
        help="Send frames as fast as possible instead of replaying at motion_fps.",
    )
    parser.add_argument("--print_joint_summary", action="store_true", default=False)
    parser.add_argument("--stats_interval", type=float, default=2.0)
    parser.add_argument(
        "--csv_positions_are_global",
        action="store_true",
        default=False,
        help="Treat CSV offsets as global positions. By default they are treated as local LuMo offsets.",
    )
    return parser


def make_socket(protocol: str, dst_ip: str, dst_port: int):
    if protocol == "udp":
        return socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((dst_ip, dst_port))
    return sock


def send_payload(sock, protocol: str, dst_ip: str, dst_port: int, payload: bytes) -> int:
    if protocol == "udp":
        return sock.sendto(payload, (dst_ip, dst_port))
    sock.sendall(payload)
    return len(payload)


def parse_frame_rate(metadata_row: List[str]) -> float | None:
    for index, value in enumerate(metadata_row[:-1]):
        if value == "FrameRate":
            try:
                return float(metadata_row[index + 1])
            except ValueError:
                return None
    return None


def parse_lumo_csv(csv_path: Path) -> Tuple[List[Dict[str, CsvBone]], float | None]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if len(rows) < 8:
        raise ValueError(f"CSV has too few rows: {csv_path}")

    frame_rate = parse_frame_rate(rows[0])
    names = rows[3]
    value_kinds = rows[5]
    components = rows[6]

    groups = []
    col = 2
    while col < len(names):
        raw_name = names[col]
        if not raw_name:
            col += 1
            continue

        bone_name = raw_name.split(":", 1)[-1]
        rotation_cols = {}
        offset_cols = {}

        while col < len(names) and names[col] == raw_name:
            if value_kinds[col] == "Rotation":
                rotation_cols[components[col]] = col
            elif value_kinds[col] == "Offset":
                offset_cols[components[col]] = col
            col += 1

        if {"X", "Y", "Z", "W"} <= rotation_cols.keys() and {"X", "Y", "Z"} <= offset_cols.keys():
            groups.append((bone_name, rotation_cols, offset_cols))

    frames: List[Dict[str, CsvBone]] = []
    for row in rows[7:]:
        if not row or not row[0].strip():
            continue

        bones_by_name: Dict[str, CsvBone] = {}
        for bone_name, rotation_cols, offset_cols in groups:
            bones_by_name[bone_name] = CsvBone(
                Name=bone_name,
                X=float(row[offset_cols["X"]]),
                Y=float(row[offset_cols["Y"]]),
                Z=float(row[offset_cols["Z"]]),
                qw=float(row[rotation_cols["W"]]),
                qx=float(row[rotation_cols["X"]]),
                qy=float(row[rotation_cols["Y"]]),
                qz=float(row[rotation_cols["Z"]]),
            )
        frames.append(bones_by_name)

    return frames, frame_rate


def maybe_print_joint_summary(frames: List[Dict[str, CsvBone]]) -> None:
    if not frames:
        print("CSV contains no frames.")
        return

    available = set(frames[0].keys())
    missing_source = [name for name in LUMO_REQUIRED_SOURCE_BONES if name not in available]
    print(f"CSV first frame has {len(available)} LuMo bones.")
    if missing_source:
        print("Missing source bones:")
        for bone_name in missing_source:
            print(f"  {bone_name}")
    else:
        print("All source bones required for streaming are present.")


def slice_frames(
    frames: List[Dict[str, CsvBone]],
    start_frame: int,
    end_frame: int | None,
) -> List[Dict[str, CsvBone]]:
    if start_frame < 1:
        raise ValueError("--start_frame is 1-based and must be >= 1")

    start_index = start_frame - 1
    end_index = end_frame if end_frame is not None else len(frames)
    selected = frames[start_index:end_index]
    if not selected:
        raise ValueError("No frames selected. Check --start_frame/--end_frame.")
    return selected


def main() -> int:
    args = build_argparser().parse_args()

    frames, csv_fps = parse_lumo_csv(args.csv_path)
    replay_fps = args.motion_fps if args.motion_fps is not None else csv_fps
    if replay_fps is None or replay_fps <= 0:
        replay_fps = 120.0

    selected_frames = slice_frames(frames, args.start_frame, args.end_frame)
    if args.print_joint_summary:
        maybe_print_joint_summary(selected_frames)

    sock = make_socket(args.protocol, args.dst_ip, args.dst_port)
    print(f"CSV path: {args.csv_path}")
    print(f"Sending to {args.dst_ip}:{args.dst_port} over {args.protocol.upper()}")
    print(f"Selected frames: {len(selected_frames)} | replay_fps: {replay_fps:g}")
    print(f"Frame payload size: {WHOLE_BODY_BYTE_SIZE} bytes (whole-body, 22 bones)")

    sent_frames = 0
    failed_frames = 0
    sent_bytes = 0
    frame_index = 0
    last_stats_log = time.time()
    frame_period = 1.0 / replay_fps

    try:
        while True:
            loop_start_time = time.time()
            bones_by_name = selected_frames[frame_index]
            human_frame = build_gmr_human_frame_from_lumo(
                bones_by_name,
                lumo_positions_are_local=not args.csv_positions_are_global,
            )

            missing = missing_whole_body_bones(human_frame)
            if missing:
                failed_frames += 1
                print(f"Skipping frame {frame_index + 1} with missing whole-body bones: {missing}")
            else:
                payload = whole_body_frame_to_bytes(human_frame)
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
                print(f"CSV send stats | ok={sent_frames} fail={failed_frames} bytes={sent_bytes}")
                last_stats_log = current_time

            frame_index += 1
            if frame_index >= len(selected_frames):
                if args.loop:
                    frame_index = 0
                else:
                    break

            if not args.no_rate_limit:
                sleep_time = frame_period - (time.time() - loop_start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        sock.close()
        print(f"Final CSV send stats | ok={sent_frames} fail={failed_frames} bytes={sent_bytes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
