import argparse
import csv
import os
import signal
import time


DEFAULT_IP = "192.168.50.150"
DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "lumo_rigid_bodies.csv")
)

RUNNING = True


def parse_id_list(value):
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_name_list(value):
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def handle_signal(signum, frame):
    del frame
    global RUNNING
    print(f"Received signal {signum}, stopping capture...")
    RUNNING = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record LuMo rigid body xyz and quaternion data to CSV."
    )
    parser.add_argument("--ip", default=DEFAULT_IP, help="LuMo server IP.")
    parser.add_argument(
        "-n",
        "--num_rigid_bodies",
        type=int,
        default=None,
        help="Number of tracked rigid bodies to record. Default: record all tracked bodies from the first valid frame.",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Comma-separated rigid body IDs to record, for example: 1,2,3.",
    )
    parser.add_argument(
        "--names",
        default=None,
        help="Comma-separated rigid body names to record. Matching is exact.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Stop after writing this many LuMo frames.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds.",
    )
    parser.add_argument(
        "--non_blocking",
        action="store_true",
        help="Use non-blocking LuMo ReceiveData(1).",
    )
    parser.add_argument(
        "--include_untracked",
        action="store_true",
        help="Also write rigid bodies whose IsTrack is false.",
    )
    parser.add_argument(
        "--status_every",
        type=int,
        default=120,
        help="Print capture status every N written frames.",
    )
    return parser.parse_args()


def rigid_key(rigid):
    return int(rigid.Id)


def filter_rigid_bodies(frame, id_filter, name_filter, include_untracked):
    rigid_bodies = []
    for rigid in frame.rigidBodys:
        if not include_untracked and not rigid.IsTrack:
            continue
        if id_filter is not None and int(rigid.Id) not in id_filter:
            continue
        if name_filter is not None and rigid.Name not in name_filter:
            continue
        rigid_bodies.append(rigid)
    rigid_bodies.sort(key=lambda item: (int(item.Id), item.Name))
    return rigid_bodies


def wait_for_selection(frame, args, id_filter, name_filter):
    candidates = filter_rigid_bodies(
        frame,
        id_filter=id_filter,
        name_filter=name_filter,
        include_untracked=args.include_untracked,
    )
    if args.num_rigid_bodies is None:
        if not candidates:
            return None
        return {rigid_key(rigid) for rigid in candidates}

    if len(candidates) < args.num_rigid_bodies:
        return None

    return {rigid_key(rigid) for rigid in candidates[: args.num_rigid_bodies]}


def write_rigid_rows(writer, frame, selected_ids, include_untracked, capture_frame_index):
    written = 0
    for rigid in frame.rigidBodys:
        if rigid_key(rigid) not in selected_ids:
            continue
        if not include_untracked and not rigid.IsTrack:
            continue

        writer.writerow(
            {
                "capture_frame_index": capture_frame_index,
                "frame_id": int(frame.FrameId),
                "timestamp": int(frame.TimeStamp),
                "camera_sync_time": int(frame.uCameraSyncTime),
                "broadcast_time": int(frame.uBroadcastTime),
                "rigid_id": int(rigid.Id),
                "rigid_name": rigid.Name,
                "is_track": bool(rigid.IsTrack),
                "x": float(rigid.X),
                "y": float(rigid.Y),
                "z": float(rigid.Z),
                "qx": float(rigid.qx),
                "qy": float(rigid.qy),
                "qz": float(rigid.qz),
                "qw": float(rigid.qw),
            }
        )
        written += 1
    return written


def main():
    args = parse_args()
    if args.num_rigid_bodies is not None and args.num_rigid_bodies <= 0:
        raise ValueError("--num_rigid_bodies must be greater than 0.")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    id_filter = parse_id_list(args.ids)
    name_filter = parse_name_list(args.names)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "capture_frame_index",
        "frame_id",
        "timestamp",
        "camera_sync_time",
        "broadcast_time",
        "rigid_id",
        "rigid_name",
        "is_track",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
    ]

    recv_flag = 1 if args.non_blocking else 0
    selected_ids = None
    if id_filter is not None and name_filter is None:
        explicit_ids = sorted(id_filter)
        if args.num_rigid_bodies is not None:
            explicit_ids = explicit_ids[: args.num_rigid_bodies]
        selected_ids = set(explicit_ids)

    written_frames = 0
    written_rows = 0
    last_wait_print_time = 0.0
    start_time = time.monotonic()

    print(f"Connecting to LuMo server: {args.ip}")
    print(f"Output CSV: {args.output}")
    if selected_ids is not None:
        print(f"Selected rigid body IDs: {sorted(selected_ids)}")

    import LuMoSDKClient

    LuMoSDKClient.Init()
    LuMoSDKClient.Connnect(args.ip)

    try:
        with open(args.output, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            while RUNNING:
                if args.duration is not None and time.monotonic() - start_time >= args.duration:
                    break
                if args.max_frames is not None and written_frames >= args.max_frames:
                    break

                frame = LuMoSDKClient.ReceiveData(recv_flag)
                if frame is None:
                    if args.non_blocking:
                        time.sleep(0.001)
                    continue

                if selected_ids is None:
                    selected_ids = wait_for_selection(frame, args, id_filter, name_filter)
                    if selected_ids is None:
                        now = time.monotonic()
                        if now - last_wait_print_time >= 1.0:
                            candidates = filter_rigid_bodies(
                                frame,
                                id_filter=id_filter,
                                name_filter=name_filter,
                                include_untracked=args.include_untracked,
                            )
                            target_text = (
                                str(args.num_rigid_bodies)
                                if args.num_rigid_bodies is not None
                                else "any"
                            )
                            print(
                                f"Waiting for {target_text} rigid bodies, "
                                f"currently found {len(candidates)}..."
                            )
                            last_wait_print_time = now
                        continue
                    print(f"Selected rigid body IDs: {sorted(selected_ids)}")

                rows_this_frame = write_rigid_rows(
                    writer,
                    frame=frame,
                    selected_ids=selected_ids,
                    include_untracked=args.include_untracked,
                    capture_frame_index=written_frames,
                )
                if rows_this_frame == 0:
                    continue

                csv_file.flush()
                written_frames += 1
                written_rows += rows_this_frame

                if args.status_every > 0 and written_frames % args.status_every == 0:
                    print(
                        f"Captured {written_frames} frames, "
                        f"{written_rows} rigid-body rows."
                    )
    finally:
        LuMoSDKClient.Close()

    print(f"Saved {written_frames} frames, {written_rows} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
