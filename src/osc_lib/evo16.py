import socket
import time

from src.osc_lib.osc_client import SimpleTCPClient



class TimeoutTCPClient(SimpleTCPClient):
    def __init__(self, address: str, port: int, timeout: float):
        super().__init__(address, port)
        self._sock.settimeout(timeout)  # <-- this line adds timeout to all socket ops


def set_preamp_gain_evo16(device, channel_gain_dict):
    """
    Set EVO-16 preamp gain over OSC protocol.
    Apply gain, disable 48V, and unmute each channel.
    Connection timeout: 3 seconds
    Per-operation timeout: 1 second
    Raises TimeoutError if any operation times out.
    """
    if device != "EVO-16":
        raise ValueError("Only EVO-16 currently supports preamp gain.")

    ip = "127.0.0.1"
    port = 9992
    connect_timeout = 5.0
    op_timeout = 2.0

    start_time = time.time()
    with TimeoutTCPClient(ip, port, timeout=connect_timeout) as tcp_client:
        if time.time() - start_time > connect_timeout:
            raise TimeoutError("Failed to connect to EVO-16 within 3 seconds")

        for ch, gain in channel_gain_dict.items():
            if gain is None:
                continue

            addr = f"/input/analogue{ch:02d}"

            for command in [
                ["set", "gain", gain],
                # ["set", "48v", 0],
                # ["set", "mute", 0]
            ]:
                try:
                    tcp_client.sendOSCMessage(addr, command)
                    start_op = time.time()
                    tcp_client.receive()
                    if time.time() - start_op > op_timeout:
                        raise TimeoutError(f"Operation timeout on channel {ch}, command: {command}")
                except socket.timeout:
                    raise TimeoutError(f"Socket timed out on channel {ch}, command: {command}")
