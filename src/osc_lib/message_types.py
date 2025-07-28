"""TODO: add docstring."""

from enum import Enum


class MessageTypes(Enum):
    """TODO: add docstring."""

    AnnounceMessage = 100
    GoodbyeMessage = 101
    OSCMessage = 47
    ApplicationMessage = 200
    OSCBundleMessage = 35
