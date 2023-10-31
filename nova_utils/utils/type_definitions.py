from enum import Enum

import numpy as np


class LabelDType(Enum):
    """Enum representing predefined label types for different annotation schemes.

        Attributes:
            DISCRETE (np.dtype): Discrete label type with fields 'from', 'to', 'id', and 'conf'.
            CONTINUOUS (np.dtype): Continuous label type with fields 'score' and 'conf'.
            FREE (np.dtype): Free-form label type with fields 'from', 'to', 'name', and 'conf'.
        """
    DISCRETE = np.dtype(
        [
            ("from", np.int32),
            ("to", np.int32),
            ("id", np.int32),
            ("conf", np.float32),
        ]
    )
    CONTINUOUS = np.dtype([("score", np.float32), ("conf", np.float32)])
    FREE = np.dtype(
        [
            ("from", np.int32),
            ("to", np.int32),
            ("name", np.object_),
            ("conf", np.float32),
        ]
    )

class SchemeType(Enum):
    """Enum representing predefined annotation schemes.

    Attributes:
        DISCRETE (int): Discrete annotation scheme with value 0.
        CONTINUOUS (int): Continuous annotation scheme with value 1.
        FREE (int): Free-form annotation scheme with value 2.
    """

    DISCRETE = 0
    CONTINUOUS = 1
    FREE = 2

## SSI Typedefs

class SSILabelDType(Enum):
    """Enum representing predefined label types for different annotation schemes as used in SSI.

    Attributes:
        DISCRETE (np.dtype): Discrete label type with fields 'from', 'to', 'id', and 'conf'.
        CONTINUOUS (np.dtype): Continuous label type with fields 'score' and 'conf'.
        FREE (np.dtype): Free-form label type with fields 'from', 'to', 'name', and 'conf'.
    """
    DISCRETE = np.dtype(
        [
            ("from", np.float64),
            ("to", np.float64),
            ("id", np.int32),
            ("conf", np.float32)
        ]
    )
    CONTINUOUS = np.dtype([("score", np.float32), ("conf", np.float32)])
    FREE = np.dtype(
        [
            ("from", np.float64),
            ("to", np.float64),
            ("name", np.object_),
            ("conf", np.float32)
        ]
    )

class SSIFileType(Enum):
    """Enum representing different file types as used by SSI.

   Attributes:
       UNDEF (int): Undefined file type with value 0.
       BINARY (int): Binary file type with value 1.
       ASCII (int): ASCII file type with value 2.
       BIN_LZ4 (int): Binary LZ4-compressed file type with value 3.
   """
    UNDEF = 0
    BINARY = 1
    ASCII = 2
    BIN_LZ4 = 3

class SSINPDataType(Enum):
    """Enum representing different data types  as used by SSI.

    Attributes:
        UNDEF (int): Undefined data type with value 0.
        SHORT (type): Short integer data type.
        USHORT (type): Unsigned short integer data type.
        INT (type): Integer data type.
        UINT (type): Unsigned integer data type.
        LONG (type): Long integer data type.
        ULONG (type): Unsigned long integer data type.
        FLOAT (type): Floating-point data type.
        DOUBLE (type): Double-precision floating-point data type.
        LDOUBLE (type): Long double-precision floating-point data type.
        BOOL (type): Boolean data type.
    """
    UNDEF = 0
    #CHAR = 1
    #UCHAR = 2
    SHORT = np.int16
    USHORT = np.uint16
    INT = np.int32
    UINT = np.uint32
    LONG = np.int64
    ULONG = np.uint64
    FLOAT = np.float32
    DOUBLE = np.float64
    LDOUBLE = np.float64
    #STRUCT = 12
    #IMAGE = 13
    BOOL = np.bool_