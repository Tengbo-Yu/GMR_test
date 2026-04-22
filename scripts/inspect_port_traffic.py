#!/usr/bin/env python3
"""Inspect live TCP/UDP traffic for a given port.

This script is aimed at quickly checking whether a port is carrying
human-readable text or binary payloads. It uses scapy for packet capture,
which means:

1. On Windows, you usually need to run it as Administrator.
2. Npcap/WinPcap-compatible capture support must be installed.

Example:
    python scripts/inspect_port_traffic.py --port 3883
    python scripts/inspect_port_traffic.py --port 3883 --iface Ethernet
    python scripts/inspect_port_traffic.py --port 3883 --max-packets 20
"""

from __future__ import annotations

import argparse
import string
import sys
from dataclasses import dataclass
from typing import Optional

try:
    from scapy.all import IP, IPv6, Raw, TCP, UDP, conf, get_if_list, sniff
except ImportError as exc:  # pragma: no cover - dependency availability varies
    print(
        "Missing dependency: scapy\n"
        "Install it with: python -m pip install scapy\n"
        "On Windows, also make sure Npcap is installed.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


PRINTABLE_ASCII = set(bytes(string.printable, "ascii"))


@dataclass
class PacketView:
    proto: str
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    payload: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture and inspect payloads on a TCP/UDP port."
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="TCP/UDP port to inspect, for example 3883.",
    )
    parser.add_argument(
        "--iface",
        default=None,
        help="Optional interface name. Omit to let scapy choose one.",
    )
    parser.add_argument(
        "--all-ifaces",
        action="store_true",
        help="Capture on all available interfaces instead of a single interface.",
    )
    parser.add_argument(
        "--max-packets",
        type=int,
        default=0,
        help="Stop after N matching packets. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=128,
        help="How many payload bytes to print per packet.",
    )
    parser.add_argument(
        "--no-hex",
        action="store_true",
        help="Hide hex dump output.",
    )
    parser.add_argument(
        "--only-with-payload",
        action="store_true",
        help="Ignore packets that carry no application payload.",
    )
    return parser.parse_args()


def get_network_layer(packet) -> Optional[object]:
    if IP in packet:
        return packet[IP]
    if IPv6 in packet:
        return packet[IPv6]
    return None


def extract_view(packet) -> Optional[PacketView]:
    network = get_network_layer(packet)
    if network is None:
        return None

    if TCP in packet:
        transport = packet[TCP]
        proto = "TCP"
    elif UDP in packet:
        transport = packet[UDP]
        proto = "UDP"
    else:
        return None

    payload = bytes(packet[Raw].load) if Raw in packet else b""
    return PacketView(
        proto=proto,
        src_ip=str(network.src),
        src_port=int(transport.sport),
        dst_ip=str(network.dst),
        dst_port=int(transport.dport),
        payload=payload,
    )


def ascii_preview(data: bytes) -> str:
    return "".join(chr(b) if 32 <= b <= 126 else "." for b in data)


def looks_like_text(data: bytes) -> bool:
    if not data:
        return False
    printable = sum((b in PRINTABLE_ASCII or b in b"\r\n\t") for b in data)
    return printable / len(data) >= 0.85


def utf8_preview(data: bytes) -> Optional[str]:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def hex_dump(data: bytes, width: int = 16) -> str:
    lines = []
    for offset in range(0, len(data), width):
        chunk = data[offset : offset + width]
        hex_part = " ".join(f"{byte:02x}" for byte in chunk)
        ascii_part = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
        lines.append(f"{offset:04x}  {hex_part:<{width * 3 - 1}}  {ascii_part}")
    return "\n".join(lines)


def print_packet(index: int, view: PacketView, payload_bytes: int, show_hex: bool) -> None:
    payload = view.payload[:payload_bytes]
    print(
        f"[{index}] {view.proto} "
        f"{view.src_ip}:{view.src_port} -> {view.dst_ip}:{view.dst_port} "
        f"payload={len(view.payload)}B"
    )

    if not view.payload:
        print("  no application payload")
        print()
        return

    text_guess = utf8_preview(payload)
    if text_guess and looks_like_text(payload):
        print("  utf8 preview:")
        for line in text_guess.splitlines() or [text_guess]:
            print(f"    {line}")
    else:
        print(f"  ascii preview: {ascii_preview(payload)}")

    if show_hex:
        print("  hex dump:")
        for line in hex_dump(payload).splitlines():
            print(f"    {line}")

    if len(view.payload) > payload_bytes:
        print(f"  ... truncated to first {payload_bytes} bytes")

    print()


def main() -> int:
    args = parse_args()

    if args.iface and args.all_ifaces:
        print("Use either --iface or --all-ifaces, not both.", file=sys.stderr)
        return 1

    if args.iface:
        available = set(get_if_list())
        if args.iface not in available:
            print("Interface not found.", file=sys.stderr)
            print("Available interfaces:", file=sys.stderr)
            for name in get_if_list():
                print(f"  {name}", file=sys.stderr)
            return 1

    capture_iface = get_if_list() if args.all_ifaces else args.iface

    capture_filter = f"tcp port {args.port} or udp port {args.port}"
    print(f"Capture filter: {capture_filter}")
    if args.all_ifaces:
        print("Interface: all available interfaces")
    else:
        print(f"Interface: {args.iface or conf.iface}")
    print("Press Ctrl+C to stop.\n")

    seen = 0

    def on_packet(packet) -> None:
        nonlocal seen
        view = extract_view(packet)
        if view is None:
            return
        if args.only_with_payload and not view.payload:
            return
        seen += 1
        print_packet(
            index=seen,
            view=view,
            payload_bytes=args.payload_bytes,
            show_hex=not args.no_hex,
        )

    try:
        sniff(
            iface=capture_iface,
            filter=capture_filter,
            prn=on_packet,
            store=False,
            count=args.max_packets if args.max_packets > 0 else 0,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "winpcap is not installed" in message.lower():
            print(
                "Packet capture backend is missing on Windows.\n"
                "Install Npcap from https://npcap.com/ and enable normal packet capture support.\n"
                "After installing Npcap, reopen the terminal and run the script again.\n"
                "Without Npcap/WinPcap, Windows cannot sniff arbitrary traffic on port 3883.",
                file=sys.stderr,
            )
            return 1
        print(f"Packet capture failed: {exc}", file=sys.stderr)
        return 1
    except PermissionError:
        print(
            "Permission denied while opening capture device.\n"
            "Try running this script in an elevated terminal.",
            file=sys.stderr,
        )
        return 1
    except OSError as exc:
        print(
            f"Packet capture failed: {exc}\n"
            "On Windows, this often means Npcap is not installed or the selected "
            "interface is unavailable.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        pass

    if seen == 0:
        print("No matching packets captured.")
        print(
            "If you expected traffic, make sure the sender is active and try a "
            "specific interface with --iface."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
