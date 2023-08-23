import numpy as np
from enum import Enum

class LabelDType(Enum):
    """Predefined label types for different annotation schemes."""
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
    """Predefined annotation schemes"""

    DISCRETE = 0
    CONTINUOUS = 1
    FREE = 2

## SSI Typedefs

class SSILabelDType(Enum):
    """Predefined label types for different annotation schemes as used in SSI."""
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
    UNDEF = 0
    BINARY = 1
    ASCII = 2
    BIN_LZ4 = 3

class SSINPDataType(Enum):
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