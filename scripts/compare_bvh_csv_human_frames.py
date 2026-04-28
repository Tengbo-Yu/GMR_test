#!/usr/bin/env python3
"""Compare BVH load_bvh_file frames with CSV build_gmr_human_frame frames."""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import io
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def install_lightweight_gmr_package() -> None:
    """Allow importing utils without importing retargeting-only dependencies."""
    if "general_motion_retargeting" not in sys.modules:
        package = types.ModuleType("general_motion_retargeting")
        package.__path__ = [str(REPO_ROOT / "general_motion_retargeting")]
        sys.modules["general_motion_retargeting"] = package

    if "general_motion_retargeting.utils" not in sys.modules:
        utils_package = types.ModuleType("general_motion_retargeting.utils")
        utils_package.__path__ = [str(REPO_ROOT / "general_motion_retargeting" / "utils")]
        sys.modules["general_motion_retargeting.utils"] = utils_package


def load_lumo_frame_helpers():
    """Load the exact helper function definitions without running script imports."""
    import general_motion_retargeting.utils.lafan_vendor.utils as lafan_utils

    source_path = REPO_ROOT / "scripts" / "lumo_live_streaming.py"
    source_ast = ast.parse(source_path.read_text(encoding="utf-8"))
    required_names = {"normalize_quat", "quat_xyzw_to_wxyz", "build_gmr_human_frame"}
    helper_ast = ast.Module(
        body=[
            node
            for node in source_ast.body
            if isinstance(node, ast.FunctionDef) and node.name in required_names
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(helper_ast)

    namespace = {
        "np": np,
        "lafan_utils": lafan_utils,
        "Dict": Dict,
        "Tuple": Tuple,
    }
    exec(compile(helper_ast, str(source_path), "exec"), namespace)
    return namespace["build_gmr_human_frame"], namespace["quat_xyzw_to_wxyz"]


def parse_lumo_csv(csv_path: Path) -> List[Dict[str, CsvBone]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if len(rows) < 8:
        raise ValueError(f"CSV has too few rows: {csv_path}")

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

        bones_by_name = {}
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

    return frames


def quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    q2 = q2 / max(np.linalg.norm(q2), 1e-12)
    dot = abs(float(np.dot(q1, q2)))
    dot = min(1.0, max(-1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def compare_frames(bvh_frames, csv_frames) -> None:
    frame_count = min(len(bvh_frames), len(csv_frames))
    common_bones = sorted(set(bvh_frames[0]).intersection(csv_frames[0]))
    missing_in_csv = sorted(set(bvh_frames[0]) - set(csv_frames[0]))
    extra_in_csv = sorted(set(csv_frames[0]) - set(bvh_frames[0]))

    per_joint = {}
    all_pos = []
    all_rot = []
    worst_pos = (-1.0, 0, "")
    worst_rot = (-1.0, 0, "")

    for bone_name in common_bones:
        pos_errors = []
        rot_errors = []
        for frame_index in range(frame_count):
            bvh_pos, bvh_quat = bvh_frames[frame_index][bone_name]
            csv_pos, csv_quat = csv_frames[frame_index][bone_name]
            pos_error = float(np.linalg.norm(np.asarray(bvh_pos) - np.asarray(csv_pos)))
            rot_error = quat_angle_deg(np.asarray(bvh_quat), np.asarray(csv_quat))
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)

            if pos_error > worst_pos[0]:
                worst_pos = (pos_error, frame_index + 1, bone_name)
            if rot_error > worst_rot[0]:
                worst_rot = (rot_error, frame_index + 1, bone_name)

        pos_array = np.asarray(pos_errors)
        rot_array = np.asarray(rot_errors)
        all_pos.extend(pos_errors)
        all_rot.extend(rot_errors)
        per_joint[bone_name] = {
            "pos_mean": float(pos_array.mean()),
            "pos_max": float(pos_array.max()),
            "rot_mean": float(rot_array.mean()),
            "rot_max": float(rot_array.max()),
        }

    all_pos_array = np.asarray(all_pos)
    all_rot_array = np.asarray(all_rot)

    print(f"BVH frames: {len(bvh_frames)}")
    print(f"CSV frames: {len(csv_frames)}")
    print(f"Compared frames: {frame_count}")
    print(f"Common bones: {len(common_bones)}")
    print(f"BVH-only bones: {', '.join(missing_in_csv) if missing_in_csv else 'none'}")
    print(f"CSV-only bones: {', '.join(extra_in_csv) if extra_in_csv else 'none'}")
    print()
    print(
        "Position error (m): "
        f"mean={all_pos_array.mean():.12g}, "
        f"p95={np.percentile(all_pos_array, 95):.12g}, "
        f"max={worst_pos[0]:.12g} at frame {worst_pos[1]} bone {worst_pos[2]}"
    )
    print(
        "Rotation error (deg): "
        f"mean={all_rot_array.mean():.12g}, "
        f"p95={np.percentile(all_rot_array, 95):.12g}, "
        f"max={worst_rot[0]:.12g} at frame {worst_rot[1]} bone {worst_rot[2]}"
    )
    print()

    print("Top position-error bones:")
    for bone_name, stats in sorted(per_joint.items(), key=lambda item: item[1]["pos_max"], reverse=True)[:10]:
        print(f"  {bone_name}: mean={stats['pos_mean']:.12g}, max={stats['pos_max']:.12g}")

    print()
    print("Top rotation-error bones:")
    for bone_name, stats in sorted(per_joint.items(), key=lambda item: item[1]["rot_max"], reverse=True)[:10]:
        print(f"  {bone_name}: mean={stats['rot_mean']:.12g}, max={stats['rot_max']:.12g}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare load_bvh_file(BVH) output with build_gmr_human_frame(CSV) output. "
            "CSV frames are not FK-converted."
        )
    )
    parser.add_argument("--bvh", type=Path, default=REPO_ROOT / "misc" / "0428_Skeleton0.bvh")
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "misc" / "0428.csv")
    parser.add_argument("--format", default="mocap", help="Format argument passed to load_bvh_file.")
    parser.add_argument(
        "--position_scale",
        type=float,
        default=0.01,
        help="Scale passed to build_gmr_human_frame; default matches lumo_live_streaming.py.",
    )
    parser.add_argument(
        "--show_bvh_loader_prints",
        action="store_true",
        help="Do not suppress debug prints inside load_bvh_file.",
    )
    args = parser.parse_args()

    install_lightweight_gmr_package()

    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    build_gmr_human_frame, quat_xyzw_to_wxyz = load_lumo_frame_helpers()

    if args.show_bvh_loader_prints:
        bvh_frames, _ = load_bvh_file(str(args.bvh), format=args.format)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            bvh_frames, _ = load_bvh_file(str(args.bvh), format=args.format)

    raw_csv_frames = parse_lumo_csv(args.csv)
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    rotation_quat_wxyz = quat_xyzw_to_wxyz(R.from_matrix(rotation_matrix).as_quat())
    csv_frames = [
        build_gmr_human_frame(
            bones_by_name,
            rotation_matrix,
            rotation_quat_wxyz,
            args.position_scale,
        )
        for bones_by_name in raw_csv_frames
    ]

    compare_frames(bvh_frames, csv_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
