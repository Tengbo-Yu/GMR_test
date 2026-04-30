#!/usr/bin/env python3
"""Print unique joint names from a LuMo exported CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print joint names from a LuMo CSV file.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path("misc/0428.csv"),
        help="Path to the LuMo CSV file. Defaults to misc/0428.csv.",
    )
    parser.add_argument(
        "--keep-prefix",
        action="store_true",
        help="Keep skeleton prefixes such as Skeleton0: in joint names.",
    )
    return parser.parse_args()


def find_name_row(rows: list[list[str]]) -> list[str]:
    for row in rows:
        if len(row) > 1 and row[1] == "Name":
            return row
    raise ValueError("Could not find CSV row whose second column is 'Name'.")


def read_joint_names(csv_path: Path, keep_prefix: bool = False) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    name_row = find_name_row(rows)
    joint_names: list[str] = []
    seen: set[str] = set()

    for raw_name in name_row[2:]:
        if not raw_name:
            continue

        joint_name = raw_name if keep_prefix else raw_name.split(":", 1)[-1]
        if joint_name in seen:
            continue

        seen.add(joint_name)
        joint_names.append(joint_name)

    return joint_names


def main() -> int:
    args = parse_args()
    joint_names = read_joint_names(args.csv_path, keep_prefix=args.keep_prefix)

    print(f"Joint count: {len(joint_names)}")
    for index, joint_name in enumerate(joint_names, start=1):
        print(f"{index:02d}. {joint_name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
