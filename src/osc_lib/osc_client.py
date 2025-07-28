"""TODO: add docstring."""

# ___MY_MODULES___
# ___MODULES___
from socket import AF_INET, SOCK_DGRAM, SOCK_STREAM, socket
from struct import pack
from uuid import getnode

from src.osc_lib.binary import packData, unpackData
from src.osc_lib.log import logger
from src.osc_lib.message_keys import ANNOUNCE_KEYS, JUCE_HEADER_KEYS, TCP_ANNOUNCE_KEYS
from src.osc_lib.message_types import MessageTypes
from src.osc_lib.node_types import NodeTypes

try:
    from collections.abc import Iterable
except ImportError:  # python 3.5
    from collections.abc import Iterable


# pip installed modules
try:
    from pythonosc.osc_bundle import OscBundle
    from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
    from pythonosc.osc_message import OscMessage
    from pythonosc.osc_message_builder import OscMessageBuilder
except ImportError:  # requires pip install
    raise ModuleNotFoundError(
        "To install the required modules use pip install python-osc"
    )

# ___CONSTANTS___
LOCAL_IP = "127.0.0.1"  # local IP
LOCAL_PORT = 9992  # UDP announce port
MAC_ADDRESS = getnode()
NODE_NAME = bytes("Client\n", "utf-8")

JUCE_PORT = LOCAL_PORT
AUDIENT_PORT = LOCAL_PORT


# ___EXCEPTIONS___
class NetworkMessageError(Exception):
    """TODO: add docstring."""

    ...


# ___CLASSES___
class UDPClient:
    """UDP client to recieve `UDPAnnounceMessage` via UDP."""

    def __init__(self, address: str = LOCAL_IP, port: int = LOCAL_PORT) -> None:
        """Initialize UDP client.

        Args:
                address: of type string IP address of server (default = 127.0.0.1 the local IP)
                port: of type int port of server (default = 9992 the juce UDP announce port)

        """
        self._address = address
        self._port = port

        self._sock = socket(AF_INET, SOCK_DGRAM)
        self._sock.bind((self._address, self._port))

        logger.debug("Connected to the UDP server successfully")

    def receiveUdpAnnounceMsg(self, timeout: int = 8) -> dict:
        """Receives a `UDPAnnounceMessage` via UDP.

        Args:
                timeout: (optional) of type int time (in seconds) to wait for UDP announce message till a TimeoutError is rasised (default = 8s as broadcast should be sent every 5s).
        """
        msg = None
        self._sock.settimeout(timeout)

        # run for a maximum of 'timeout' seconds between each recieved message

        # if a message was recieved ...
        while True:
            # listen for a message of 1024 bytes
            msg = self._sock.recv(1024)

            # unpack the message type
            if msg is not None:
                if msg[0] == MessageTypes.AnnounceMessage.value:
                    try:
                        umsg = unpackData(ANNOUNCE_KEYS, msg)
                        logger.debug("The UDP client recieved an announce message")
                        return umsg
                    except Exception:
                        raise NetworkMessageError(
                            "The recieved announce message is not in the correct format."
                        )
                else:
                    logger.debug(
                        "Message recieved from UDP client was not an announce message, waiting for next message ..."
                    )
                    continue

    def __enter__(self):
        """TODO: add docstring."""
        return self

    def __exit__(self, *args):
        """TODO: add docstring."""
        self._sock.close()
        del self._sock
        logger.debug("Disconnected from the UDP server")

    def __del__(self):
        """TODO: add docstring."""
        if hasattr(self, "_sock"):
            self.__exit__()


class TCPClient:
    """OSC client to send :class:`OscMessage` via TCP."""

    def __init__(self, port: int, address: str = LOCAL_IP) -> None:
        """Initialize TCP client.

                As this is TCP it will make an attempt to connect to the
                given server at ip:port until the object is destroyed.

        Args:
                port: of type int port of TCP server
                address: of type string IP address of server (default = 127.0.0.1 the local IP)

        """
        self._port = port
        self._address = address

        self._sock = socket(AF_INET, SOCK_STREAM)

        self._sock.connect((self._address, self._port))

        self._my_port = self._sock.getsockname()[1]

    def send(self, msg: bytes = b"") -> None:
        """Send a message via TCP.

        Args:
                msg: of type 'bytes' message to be sent.
        """
        header = self.buildTCPHeader(len(msg))
        self._sock.send(header + msg)

    def receive(
        self, timeout: int = 5
    ) -> tuple[MessageTypes, dict | bytes | OscMessage | OscBundle]:
        """Receive a message of type bytes via TCP.

        Args:
                timeout: (optional) of type int time (in seconds) to wait for TCP message till a TimeoutError is raised (default = 30 seconds)
        Returns:
                tuple of types MessageTypes, dict OscMessage or OscBundle.

        """
        msg = None

        # run for a maximum of 'timeout' seconds between received messages
        self._sock.settimeout(timeout)

        while True:
            # wait for a message of upto 10kB - currently our max length is < 2kB
            msg = self._sock.recv(1024 * 10)

            # if a message was recieved ...
            if msg is not None:
                if msg[16] == MessageTypes.AnnounceMessage.value:
                    logger.debug("The TCP client recieved an announce message")
                    return MessageTypes.AnnounceMessage, self.decodeAnnounceMsg(msg)

                elif msg[16] == MessageTypes.OSCMessage.value:
                    logger.debug("The TCP client recieved an osc message")
                    return MessageTypes.OSCBundleMessage, self.decodeOSCMsg(msg)

                elif msg[16] == MessageTypes.ApplicationMessage.value:
                    logger.debug(
                        "The TCP client recieved an application message (not currently created a deserialisation for this message type)"
                    )
                    return MessageTypes.ApplicationMessage, msg

                elif msg[16] == MessageTypes.OSCBundleMessage.value:
                    logger.debug(
                        "The TCP client recieved an OSC bundle message (not currently created a deserialisation for this message type)"
                    )
                    logger.warning(
                        "Received bundle messages are not currently supported"
                    )
                    return MessageTypes.OSCBundleMessage, msg

                elif msg[16] == MessageTypes.GoodbyeMessage.value:
                    logger.debug(
                        "The TCP client recieved a goodbye message (object will self destruct)"
                    )
                    del self
                    return MessageTypes.GoodbyeMessage, msg

                else:
                    logger.debug(
                        "Message recieved from TCP client was not a recognised message type, waiting for next message ..."
                    )

    def clear(self):
        """Clear the input buffer."""
        self._sock.recv(1024)

    def buildTCPHeader(self, message_size: int) -> bytes:
        """Build the TCP juce and audient header from arguments and return the joint header.

        Args:
                message_size: of type int message byte size
                port: of type int juce and audient port.
        """
        juce_header = pack("<II", JUCE_PORT, message_size + 8)
        audient_header = pack(">II", message_size, AUDIENT_PORT)
        return juce_header + audient_header

    def buildOSCMessage(
        self, address: str, value: int | float | bytes | str | bool | tuple | list
    ) -> OscMessage:
        """Build :class:`OscMessage` from arguments and return the message.

        Args:
                address: of type string OSC address the message shall go to
                value: list of one or more arguments to be added to the message.
        """
        builder = OscMessageBuilder(address=address)
        if value is None:
            values = []
        elif not isinstance(value, Iterable) or isinstance(value, str | bytes):
            values = [value]
        else:
            values = value
        for val in values:
            builder.add_arg(val)  # type: ignore

        return builder.build()

    def buildOSCBundle(self, *osc_messages: OscBundle) -> OscBundle:
        """Build :class:`OscBundle` from arguments and return the message.

        Args:
                osc_messages: args of type *OscMessage.
        """
        builder = OscBundleBuilder(IMMEDIATELY)
        for msg in osc_messages:
            builder.add_content(msg)
        return builder.build()

    def buildInteractionMessage(self, isannounce: bool) -> bytes:
        """Build a simple interaction message i.e. announce or goodbye message.

        Args:
                isannounce: of type bool, true if message is announce or false if goodbye.
        """
        announce = {
            "message_type": MessageTypes.AnnounceMessage.value
            if isannounce
            else MessageTypes.GoodbyeMessage.value,
            "protocol_version": 1,
            "reserved": 0,
            "sequence_number": 0,
            "timestamp": 0.0,
            "reserved2": 0,
            "udp_discovery_port": self._my_port,
            "tcp_port": self._my_port,
            "up_time": 0,
            "node_type": NodeTypes.Client.value,
            "node_id": MAC_ADDRESS,
            "string_bytes": len(NODE_NAME),
            "node_name": NODE_NAME,
            "reply_request": False,
        }
        return packData(ANNOUNCE_KEYS, *tuple(announce.values()))

    def decodeOSCMsg(self, msg: bytes) -> OscBundle:
        """Decode an osc message (or multiple messages) and convert to OscBundle.

        Args:
                msg: of type bytes to unpack.
        """
        msg_start = 0
        msg_end = 0
        osc_bundle_builder = OscBundleBuilder(IMMEDIATELY)

        while len(msg) > msg_end:
            umsg = unpackData(JUCE_HEADER_KEYS, msg[msg_start : 16 + msg_start])
            msg_end += umsg["j_bytes"] + 8

            osc_bundle_builder.add_content(OscBundle(msg[msg_start + 16 : msg_end]))

            msg_start = msg_end

        return osc_bundle_builder.build()

    def decodeAnnounceMsg(self, msg: bytes) -> dict:
        """Decode an `OSCMessage`.

        Args:
                msg: of type bytes to unpack.
        """
        umsg = unpackData(TCP_ANNOUNCE_KEYS, msg)
        umsg["node_name"] = umsg["node_name"].decode("utf-8").rstrip("\x00")

        return umsg

    def sendAnnounce(self) -> bool:
        """Send a simple announce message."""
        announce = self.buildInteractionMessage(isannounce=True)
        self.send(announce)

        logger.debug("The TCP client sent an announce message")

        return self.receive()[0] == MessageTypes.AnnounceMessage

    def sendGoodbye(self) -> None:
        """Send a simple goodbye message."""
        goodbye = self.buildInteractionMessage(isannounce=False)
        self.send(goodbye)

        logger.debug("The TCP client sent a goodbye message")

    def __enter__(self):
        """TODO: add docstring."""
        return self

    def __exit__(self, *args):
        """TODO: add docstring."""
        self.sendGoodbye()
        self._sock.close()
        del self._sock
        logger.debug("Disconnected from the TCP server")

    def __del__(self):
        """TODO: add docstring."""
        if hasattr(self, "_sock"):
            self.__exit__()


class SimpleTCPClient(TCPClient):
    """Simple OSC client that automatically builds :class:`OscMessage` from arguments."""

    def __init__(
        self, address: str = LOCAL_IP, udp_port: int = LOCAL_PORT, *arg, **kw
    ) -> None:
        """Connect to the TCP server using UDPClient.

        Args:
                address: of type string IP address of server (default = 127.0.0.1 the local IP)
                udp_port: of type int port of udp server (default = 9992 the juce UDP announce port)

        """
        with UDPClient(address, udp_port) as udp_client:
            try:
                udp_announce_msg = udp_client.receiveUdpAnnounceMsg()
            except TimeoutError:
                raise NetworkMessageError(
                    "Failed to connect to the device, please make sure the device is connected and the EVO app is open"
                )

        super().__init__(udp_announce_msg["tcp_port"], address, *arg, **kw)

        # send announce and check response before continuing

        if self.sendAnnounce():
            logger.debug("Connected to the TCP server successfully")
        else:
            raise NetworkMessageError(
                "Expected an announce message but received a message of a different type"
            )

    def sendOSCMessage(
        self, address: str, value: int | float | bytes | str | bool | tuple | list
    ) -> None:
        """Send a simple OSC message with the juce and audient header.

        Args:
                address: of type string OSC address the message shall go to
                value: list of one or more arguments to be added to the message.
        """
        osc = self.buildOSCMessage(address, value)
        self.send(osc.dgram)

        logger.debug("The TCP client sent an OSC message")

    def sendOSCBundle(self, *msgs) -> None:
        """Send a simple OSC bundle with the juce and audient header.

        !!! NOT CURRENTLY WORKING (waiting on implementation in EVO app) !!!
        Args:
                msgs: *args of type dict in the format:
                        {'address': "/input/analogue01", 'value': ["set", "gain", -8.0]},
                        {'address': "/input/analogue02", 'value': ["set", "gain", -8.0]},
                        ...
        """
        osc_msgs = []
        for msg in msgs:
            osc_msgs.append(self.buildOSCMessage(msg["address"], msg["value"]))

        osc_bundle = self.buildOSCBundle(*osc_msgs)
        self.send(osc_bundle.dgram)

        logger.debug("The TCP client sent an OSC bundle message")
