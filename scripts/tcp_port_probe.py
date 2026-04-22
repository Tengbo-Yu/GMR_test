#!/usr/bin/env python3
"""Connect to a TCP port and print received bytes.

Useful when a local server only starts sending data after a client connects.
This is a good fit for checking whether a VRPN-like service on port 3883
actually emits payloads once a TCP session is established.
"""

from __future__ import annotations

import argparse
import socket
import string
import sys
from typing import Optional


PRINTABLE_ASCII = set(bytes(string.printable, "ascii"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to a TCP server and dump received payloads."
    )
    parser.add_argument("--host", required=True, help="Server IP or hostname.")
    parser.add_argument("--port", type=int, required=True, help="Server TCP port.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Socket timeout in seconds.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="How long to keep reading before exit.",
    )
    parser.add_argument(
        "--recv-size",
        type=int,
        default=4096,
        help="Bytes per recv call.",
    )
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=128,
        help="How many bytes to print from each received chunk.",
    )
    return parser.parse_args()


def ascii_preview(data: bytes) -> str:
    return "".join(chr(b) if 32 <= b <= 126 else "." for b in data)


def looks_like_text(data: bytes) -> bool:
    if not data:
        return False
    printable = sum((b in PRINTABLE_ASCII or b in b"\r\n\t") for b in data)
    return printable / len(data) >= 0.85


def hex_dump(data: bytes, width: int = 16) -> str:
    lines = []
    for offset in range(0, len(data), width):
        chunk = data[offset : offset + width]
        hex_part = " ".join(f"{byte:02x}" for byte in chunk)
        ascii_part = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
        lines.append(f"{offset:04x}  {hex_part:<{width * 3 - 1}}  {ascii_part}")
    return "\n".join(lines)


def try_decode_utf8(data: bytes) -> Optional[str]:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def main() -> int:
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.timeout)

    print(f"Connecting to {args.host}:{args.port} ...")
    try:
        sock.connect((args.host, args.port))
    except OSError as exc:
        print(f"Connect failed: {exc}", file=sys.stderr)
        return 1

    print("Connected. Waiting for payloads...\n")
    sock.settimeout(min(args.timeout, 1.0))

    total_bytes = 0
    chunk_index = 0

    try:
        deadline = socket.getdefaulttimeout()
        # Use monotonic timing without importing time into the main path until needed.
        import time

        end_time = time.monotonic() + args.duration
        while time.monotonic() < end_time:
            try:
                data = sock.recv(args.recv_size)
            except socket.timeout:
                continue

            if not data:
                print("Server closed the connection.")
                break

            chunk_index += 1
            total_bytes += len(data)
            preview = data[: args.payload_bytes]

            print(f"[{chunk_index}] received {len(data)} bytes")
            decoded = try_decode_utf8(preview)
            if decoded and looks_like_text(preview):
                print("  utf8 preview:")
                for line in decoded.splitlines() or [decoded]:
                    print(f"    {line}")
            else:
                print(f"  ascii preview: {ascii_preview(preview)}")

            print("  hex dump:")
            for line in hex_dump(preview).splitlines():
                print(f"    {line}")
            if len(data) > args.payload_bytes:
                print(f"  ... truncated to first {args.payload_bytes} bytes")
            print()
    finally:
        sock.close()

    if chunk_index == 0:
        print("No payload received during the probe window.")
        print(
            "This usually means the service is only listening, waiting for a "
            "different handshake, or not streaming on this TCP port yet."
        )
    else:
        print(f"Done. Received {total_bytes} bytes across {chunk_index} chunks.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
