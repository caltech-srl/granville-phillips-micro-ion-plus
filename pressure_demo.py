from granville_phillips_392_logger import GranvillePhillips392, parse_args, print_serial_ports, GaugeProtocolError
import sys

args = parse_args()

def main():
    if args.list_ports:
        print_serial_ports()
        return

    if not args.port:
        print("Error: --port is required unless --list-ports is used.", file=sys.stderr)
        return

    if args.interval <= 0:
        print("Error: --interval must be greater than zero.", file=sys.stderr)

    gauge = GranvillePhillips392(
        port=args.port,
        address=args.address,
        baudrate=args.baud,
        timeout=args.timeout,
    )

    gauge.connect()
    try:
        gauge.read_pressure()
    except GaugeProtocolError as exc:
        print(f"read error: {exc}", file=sys.stderr)

    gauge.close()

if __name__ == "__main__":
    main()