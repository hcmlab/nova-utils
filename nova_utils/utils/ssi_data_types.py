"""SSI data type definitions

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    14.9.2023

"""

from enum import Enum

import numpy as np


class FileTypes(Enum):
    """
    Enum representing different file types.

    This enum defines constants for various file types, including UNDEF, BINARY, ASCII, and BIN_LZ4.

    Attributes:
        UNDEF (int): Undefined file type.
        BINARY (int): Binary file type.
        ASCII (int): ASCII file type.
        BIN_LZ4 (int): Binary LZ4 compressed file type.

    """

    UNDEF = 0
    BINARY = 1
    ASCII = 2
    BIN_LZ4 = 3


class NPDataTypes(Enum):
    """
    Enum representing different NumPy data types.

    This enum defines constants for various NumPy data types, including UNDEF, SHORT, USHORT, INT, UINT, LONG, ULONG, FLOAT, DOUBLE, and BOOL.

    Attributes:
        UNDEF (int): Undefined data type.
        SHORT (numpy.int16): 16-bit integer data type.
        USHORT (numpy.uint16): Unsigned 16-bit integer data type.
        INT (numpy.int32): 32-bit integer data type.
        UINT (numpy.uint32): Unsigned 32-bit integer data type.
        LONG (numpy.int64): 64-bit integer data type.
        ULONG (numpy.uint64): Unsigned 64-bit integer data type.
        FLOAT (numpy.float32): 32-bit floating-point data type.
        DOUBLE (numpy.float64): 64-bit floating-point data type.
        BOOL (numpy.bool_): Boolean data type.

    """

    UNDEF = 0
    # CHAR = 1
    # UCHAR = 2
    SHORT = np.int16
    USHORT = np.uint16
    INT = np.int32
    UINT = np.uint32
    LONG = np.int64
    ULONG = np.uint64
    FLOAT = np.float32
    DOUBLE = np.float64
    LDOUBLE = np.float64
    # STRUCT = 12
    # IMAGE = 13
    BOOL = np.bool_


"""Helper"""


def string_to_enum(enum, string):
    """
    Convert a string to an enum value.

    This function takes an enum and a string and returns the corresponding enum value. If the string does not match any enum value, a ValueError is raised.

    Args:
        enum (Enum): The enum to search in.
        string (str): The string to convert to an enum value.

    Returns:
        Enum: The enum value corresponding to the input string.

    Raises:
        ValueError: If the input string does not match any enum value in the specified enum.

    """
    for e in enum:
        if e.name == string:
            return e
    raise ValueError("{} not part of enumeration  {}".format(string, enum))
