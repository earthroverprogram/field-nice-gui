from pathlib import Path

from log import logger
from osc_client import SimpleTCPClient


def get_channel_address(channel: int) -> str:
    return f"/input/analogue{channel:02d}"


def set_all_channels(tcp_client, config_lines):
    for line in config_lines:
        # Remove inline comments and strip whitespace
        logger.info(f"Processing: {line}")
        line = line.split("#", 1)[0].strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 2 or len(parts) > 4:
            logger.warning(f"Ignoring line with unexpected format: {line}")
            continue

        try:
            ch = int(parts[0])
            gain = float(parts[1])
            mute = int(parts[2]) if len(parts) >= 3 else None
            v48 = int(parts[3]) if len(parts) == 4 else None
        except ValueError:
            logger.warning(f"Ignoring line with invalid values: {line}")
            continue

        addr = get_channel_address(ch)

        tcp_client.sendOSCMessage(addr, ["set", "gain", gain])
        tcp_client.receive()

        if mute is not None:
            tcp_client.sendOSCMessage(addr, ["set", "mute", mute])
            tcp_client.receive()

        if v48 is not None:
            tcp_client.sendOSCMessage(addr, ["set", "48v", v48])
            tcp_client.receive()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apply EVO 16 Mic Preamp settings via OSC")
    parser.add_argument("--ip", default="127.0.0.1", help="Target IP address")
    parser.add_argument("--port", type=int, default=9992, help="Target TCP port")
    parser.add_argument("--config", required=True, help="Path to config file (required)")

    args = parser.parse_args()
    config_lines = Path(args.config).read_text().splitlines()
    with SimpleTCPClient(args.ip, args.port) as tcp_client:
        set_all_channels(tcp_client, config_lines)
        print(f"Settings applied from: {args.config}")


if __name__ == "__main__":
    logger.setLevel(1)
    main()
