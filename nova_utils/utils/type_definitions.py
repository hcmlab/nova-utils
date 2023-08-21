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

class SchemeType(Enum):
    """Predefined annotation schemes"""

    DISCRETE = 0
    CONTINUOUS = 1
    FREE = 2