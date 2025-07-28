"""TODO: add docstring."""

# ___MODULES___
from struct import calcsize, pack, unpack


# ___FUNCTIONS___
# Deserialise
def unpackData(keys: list, msg: bytes) -> dict:
    """Unpack data using the message in bytes and a list tuples with the key, type and endianness.

    Args:
            keys: list of tuples [('header', 'I', '!'), ('message_type', 'B', '!')]
            msg: of type bytes message recived from the server.
    """
    previous_endian, current_endian = keys[0][2], keys[0][2]
    unpack_str = ""
    unpack_data = ()
    current_bytes, previous_bytes = 0, 0

    for i in range(len(keys)):
        current_endian = keys[i][2]

        # Have to unpack
        if (current_endian != previous_endian) or (keys[i][1] == "s"):
            current_bytes = calcsize(previous_endian + unpack_str) + previous_bytes
            unpack_data += unpack(
                previous_endian + unpack_str, msg[previous_bytes:current_bytes]
            )
            if keys[i][1] == "s":
                unpack_str = str(unpack_data[-1]) + keys[i][1]  # type: ignore
            else:
                unpack_str = keys[i][1]
            previous_bytes = current_bytes

        # Don't have to unpack
        elif current_endian == previous_endian:
            unpack_str += keys[i][1]

        previous_endian = current_endian

    current_bytes = calcsize(current_endian + unpack_str) + previous_bytes
    unpack_data += unpack(current_endian + unpack_str, msg[previous_bytes:])

    labels = [label[0] for label in keys]
    unpack_dict = dict(zip(labels, unpack_data))

    return unpack_dict


# Serialise
def packData(keys, *msg) -> bytes:
    """Pack data using the message as a list of values and their keys.

    Args:
            keys: list of tuples [('header', 'I', '!'), ('message_type', 'B', '!')]
            *msg: of type args tuple message with values to pack.
    """
    previous_endian, current_endian = keys[0][2], keys[0][2]
    pack_str = ""
    pack_data = b""
    current_index, previous_index = 0, 0

    for i in range(len(keys)):
        current_endian = keys[i][2]
        current_index = i

        # Have to pack
        if (current_endian != previous_endian) or (keys[i][1] == "s"):
            pack_data += pack(
                previous_endian + pack_str, *msg[previous_index:current_index]
            )
            if keys[i][1] == "s":
                pack_str = str(msg[i - 1]) + keys[i][1]
            else:
                pack_str = keys[i][1]
            previous_index = current_index

        # Don't have to pack
        elif current_endian == previous_endian:
            pack_str += keys[i][1]

        previous_endian = current_endian

    pack_data += pack(current_endian + pack_str, *msg[previous_index:])

    return pack_data
