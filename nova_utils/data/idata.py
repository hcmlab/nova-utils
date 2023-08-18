from abc import ABC, abstractmethod
import numpy as np
from nova_utils.data.meta_data import MetaData
from typing import Optional, Any, Type

# class DataType(Enum):
#     """Enumeration of possible data types."""
#
#     # Dynamic signals
#     SIGNAL_VIDEO = 0
#     SIGNAL_AUDIO = 1
#     SIGNAL_FEATURE = 2
#
#     # Static signals
#     SIGNAL_IMAGE = 3
#
#     # Annotations
#     ANNOTATION_DISCRETE = 4
#     ANNOTATION_CONTINUOUS = 5
#     ANNOTATION_FREE = 6
#     ANNOTATION_POINT = 7
#     ANNOTATION_DISCRETE_POLYGON = 8



# DATA
class IData(ABC):
    """Abstract base class for all data types."""

    def __init__(self, data: np.ndarray, dataset: str = None, role: str = None, session: str = None):
        """
        Initialize data.

        Args:
            data (np.ndarray, optional): The data array. Defaults to None.
        """
        self._data = data
        self.meta_data = MetaData(dataset, role, session)


    @property
    def data(self) -> np.ndarray:
        """
        Get the full data array.

        Returns:
            np.ndarray: The data array.
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


class IStaticData(IData):
    """Abstract base class for static data."""
    ...


class IDynamicData(IData):
    """Abstract base class for dynamic data."""

    @abstractmethod
    def sample_from_interval(self, start: int, end: int) -> np.ndarray:
        """
        Sample data from the specified time interval.

        Args:
            start (int): The start time of the interval in milliseconds.
            end (int): The end time of the interval in milliseconds.

        Returns:
            np.ndarray: The sampled data within the specified interval.
        """
