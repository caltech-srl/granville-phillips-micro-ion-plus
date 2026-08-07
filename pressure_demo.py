from granville_phillips_392_logger import (
    GranvillePhillips392,
    GaugeProtocolError,
)
from serial.tools import list_ports
import serial
import sys


# Configuration
PORT = None          # e.g. "COM5" or "/dev/ttyUSB0"; None = auto-select
ADDRESS = 1          # Gauge address, 0x0 through 0xF
BAUD = 19200
TIMEOUT = 1.0


def select_serial_port():
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
        print(
            f"  {index}. {port.device}: "
            f"{port.description or 'Unknown device'}"
        )

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


def main():
    port = PORT

    # If no port was configured, automatically select one.
    if port is None:
        port = select_serial_port()

    if port is None:
        return 2

    gauge = GranvillePhillips392(
        port=port,
        address=ADDRESS,
        baudrate=BAUD,
        timeout=TIMEOUT,
    )

    try:
        gauge.connect()
        print(
            f"Connected to {port}; "
            f"address={ADDRESS:02X}, "
            f"baud={BAUD}, "
            f"unit={gauge.unit}"
        )

        pressure = gauge.read_pressure()
        print(f"Pressure: {pressure:.9g} {gauge.unit}")

    except GaugeProtocolError as exc:
        print(f"Read error: {exc}", file=sys.stderr)
        return 1

    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 1

    finally:
        gauge.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

