from enum import Enum
import numpy as np

class FileTypes(Enum):
        UNDEF = 0
        BINARY = 1
        ASCII = 2
        BIN_LZ4 = 3

class NPDataTypes(Enum):
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

'''Helper'''
def string_to_enum(enum, string):
    for e in enum:
        if e.name == string:
            return e
    raise ValueError('{} not part of enumeration  {}'.format(string, enum))
