"""TODO: add docstring."""

ANNOUNCE_KEYS = [
    ("message_type", "B", "!"),  # |   u8   |   big endian  |
    ("protocol_version", "I", "!"),  # |  u32   |       ''      |
    ("reserved", "B", "!"),  # |   u8   |       ''      |
    ("sequence_number", "I", "!"),  # |  u32   |       ''      |
    ("timestamp", "d", "!"),  # | double |       ''      |
    ("reserved_2", "I", "!"),  # |  u32   |       ''      |
    ("udp_discovery_port", "I", "!"),  # |  u32   |       ''      |
    ("tcp_port", "I", "!"),  # |  u32   |       ''      |
    ("up_time", "Q", "!"),  # |  u64   |       ''      |
    ("node_type", "I", "<"),  # |  u32   | little endian |
    ("node_id", "Q", "!"),  # |  u64   |   big endian  |
    ("node_name_length", "I", "!"),  # | string |       ''      |
    ("node_name", "s", "!"),  # | string |       ''      |
    ("reply_request", "?", "!"),  # |  bool  |       ''      |
]

JUCE_HEADER_KEYS = [
    ("juce_header", "I", "<"),  # |  u32   | little endian |
    ("j_bytes", "I", "<"),  # |  u32   |       ''      |
    ("a_bytes", "I", "!"),  # |  u32   |   big endian  |
    ("audient_header", "I", "!"),  # |  u32   |       ''      |
]

TCP_ANNOUNCE_KEYS = JUCE_HEADER_KEYS + ANNOUNCE_KEYS
