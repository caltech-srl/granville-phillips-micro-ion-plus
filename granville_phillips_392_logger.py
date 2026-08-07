#!/usr/bin/env python3
"""
Granville-Phillips 392 Micro-Ion Plus pressure logger.

Reads pressure every 30 seconds (configurable) and appends:
    timestamp, pressure, unit
to a CSV file.

Install:
    python -m pip install pyserial

Examples:
    Windows:
        python granville_phillips_392_logger.py --port COM5

    Linux:
        python granville_phillips_392_logger.py --port /dev/ttyUSB0

    Auto-select/prompt for a serial port:
        python granville_phillips_392_logger.py

    List serial ports:
        python granville_phillips_392_logger.py --list-ports

Important:
    This script assumes the external RS-485/RS-232 converter performs
    automatic half-duplex transmit/receive direction switching.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import serial
from serial.tools import list_ports


FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$"
)


class GaugeError(RuntimeError):
    """Base class for gauge communication errors."""


class GaugeProtocolError(GaugeError):
    """The gauge returned no response or an invalid response."""


class GranvillePhillips392:
    def __init__(
        self,
        port: str,
        address: int = 1,
        baudrate: int = 19200,
        timeout: float = 1.0,
    ) -> None:
        if not 0 <= address <= 0xF:
            raise ValueError("Address must be between 0 and F.")

        self.port_name = port
        self.address = f"{address:02X}"
        self.baudrate = baudrate
        self.timeout = timeout
        self.port: Optional[serial.Serial] = None
        self.unit = "UNKNOWN"

    def connect(self) -> None:
        self.close()
        self.port = serial.Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=self.timeout,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )

        # Let the converter and line settle, then query the configured unit.
        time.sleep(0.1)
        self.unit = self.query("RU").strip().upper()

    def close(self) -> None:
        if self.port is not None:
            try:
                self.port.close()
            finally:
                self.port = None

    def query(self, command: str) -> str:
        """
        Send one gauge command and return the response data field.

        Command framing:
            #<one-digit hex address><command><CR>

        CR is sent without LF. Converter echo, if present, is ignored.
        """
        if self.port is None or not self.port.is_open:
            raise serial.SerialException("Serial port is not open.")

        request_text = f"#{self.address}{command}"
        request = (request_text + "\r").encode("ascii")

        self.port.reset_input_buffer()
        self.port.write(request)
        self.port.flush()

        deadline = time.monotonic() + self.timeout
        expected_ok_prefix = f"*{self.address}"
        expected_error_prefix = f"?{self.address}"

        while time.monotonic() < deadline:
            raw = self.port.read_until(b"\r", size=128)
            if not raw:
                break

            text = raw.decode("ascii", errors="replace").strip("\r\n")

            # Some RS-232/RS-485 converters echo the transmitted request.
            if text == request_text:
                continue

            if text.startswith(expected_error_prefix):
                raise GaugeProtocolError(f"Gauge error response: {text!r}")

            if text.startswith(expected_ok_prefix):
                return text[len(expected_ok_prefix):].strip()

            # Ignore unrelated/partial data and continue until timeout.

        raise GaugeProtocolError(
            f"No valid reply to {request_text!r} within "
            f"{self.timeout:.2f} seconds."
        )

    def read_pressure(self) -> float:
        response = self.query("RD")

        if not FLOAT_PATTERN.fullmatch(response):
            raise GaugeProtocolError(
                f"Pressure response is not numeric: {response!r}"
            )

        pressure = float(response)

        # The module uses 9.99E+09 when it cannot provide valid pressure.
        if math.isclose(pressure, 9.99e9, rel_tol=0.0, abs_tol=1.0):
            raise GaugeProtocolError(
                "Gauge reported 9.99E+09 (no valid pressure available)."
            )

        return pressure


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat()

def local_timestamp_path(usage=None) -> str:
    if usage == "example":
        return "YYYY-MM-DD"
    return datetime.now().astimezone().strftime("%Y-%m-%d")


def append_csv(
    csv_path: Path,
    timestamp: str,
    pressure: Optional[float],
    unit: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if needs_header:
            writer.writerow(["timestamp", "pressure", "unit"])

        pressure_text = "" if pressure is None else f"{pressure:.9g}"
        writer.writerow([timestamp, pressure_text, unit])
        file.flush()


def print_serial_ports() -> None:
    ports = list(list_ports.comports())

    if not ports:
        print("No serial ports found.")
        return

    for port in ports:
        print(f"{port.device}: {port.description or 'Unknown device'}")


def select_serial_port() -> Optional[str]:
    """Automatically select one serial port or prompt when several exist."""
    ports = list(list_ports.comports())

    if not ports:
        print("Error: no serial ports found.", file=sys.stderr)
        return None

    if len(ports) == 1:
        port = ports[0]
        print(
            f"Using the only detected serial port: "
            f"{port.device} ({port.description or 'Unknown device'})"
        )
        return port.device

    print("Multiple serial ports found:")
    for index, port in enumerate(ports, start=1):
        print(f"  {index}. {port.device}: {port.description or 'Unknown device'}")

    while True:
        try:
            choice = input(f"Select a port [1-{len(ports)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nPort selection cancelled.", file=sys.stderr)
            return None

        try:
            selection = int(choice)
        except ValueError:
            print(f"Please enter a number from 1 to {len(ports)}.")
            continue

        if 1 <= selection <= len(ports):
            return ports[selection - 1].device

        print(f"Please enter a number from 1 to {len(ports)}.")


def current_csv_path(base_dir: Path) -> Path:
    date = local_timestamp_path()
    return base_dir / f"pressure_log_{date}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Log pressure from a Granville-Phillips 392 Micro-Ion Plus "
            "module to CSV."
        )
    )
    parser.add_argument(
        "--port",
        help="Serial port, such as COM5 or /dev/ttyUSB0.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(".data/"),
        help="Directory where daily CSV files are stored (default: ./data/).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between readings (default: 10).",
    )
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 16),
        default=1,
        help="Gauge hex address from 0 to F (default: 1).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=19200,
        choices=[1200, 2400, 4800, 9600, 19200, 38400],
        help="Serial baud rate (default: 19200).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Reply timeout in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=5.0,
        help="Seconds before reconnecting after an error (default: 5).",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List serial ports and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_ports:
        print_serial_ports()
        return 0

    if not args.port:
        args.port = select_serial_port()
        if not args.port:
            return 2

    if args.interval <= 0:
        print("Error: --interval must be greater than zero.", file=sys.stderr)
        return 2

    gauge = GranvillePhillips392(
        port=args.port,
        address=args.address,
        baudrate=args.baud,
        timeout=args.timeout,
    )

    next_read = time.monotonic()

    try:
        while True:
            # Keep trying to establish/re-establish communication.
            if gauge.port is None or not gauge.port.is_open:
                try:
                    gauge.connect()
                    print(
                        f"Connected to {args.port}; "
                        f"address={args.address:X}, baud={args.baud}, "
                        f"unit={gauge.unit}"
                    )
                    next_read = time.monotonic()
                except (serial.SerialException, GaugeError) as exc:
                    print(f"{local_timestamp()} | connect error: {exc}", file=sys.stderr)
                    gauge.close()
                    time.sleep(args.reconnect_delay)
                    continue

            sleep_seconds = next_read - time.monotonic()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            timestamp = local_timestamp()

            try:
                pressure = gauge.read_pressure()
                csv_path = current_csv_path(args.csv)
                append_csv(csv_path, timestamp, pressure, gauge.unit)
                print(
                    f"{timestamp} | {pressure:.9g} {gauge.unit} "
                    f"| appended to {csv_path}"
                )
            except GaugeProtocolError as exc:
                # Preserve the 30-second sampling record with a blank pressure.
                csv_path = current_csv_path(args.csv)
                append_csv(csv_path, timestamp, None, gauge.unit)
                print(f"{timestamp} | read error: {exc}", file=sys.stderr)
            except serial.SerialException as exc:
                print(f"{timestamp} | serial error: {exc}", file=sys.stderr)
                gauge.close()

            next_read += args.interval

            # If delayed by more than one interval, resume from now rather
            # than rapidly issuing multiple catch-up requests.
            now = time.monotonic()
            if next_read < now - args.interval:
                next_read = now + args.interval

    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    finally:
        gauge.close()


if __name__ == "__main__":
    raise SystemExit(main())