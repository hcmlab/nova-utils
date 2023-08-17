from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any, Union

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

# METADATA
# METADATA
class GeneralMetaData:
    def __init__(self, dataset: str = None, role: str = None, session: str = None):
        self.dataset = dataset
        self.role = role
        self.session = session


class CommonMetaHandler:
    def __int__(self, handler_type: type):
        self.handler_type = handler_type


# DATA
class IData(ABC):
    """Abstract base class for all data types."""

    def __init__(self, data: np.ndarray):
        """
        Initialize data.

        Args:
            data (np.ndarray, optional): The data array. Defaults to None.
        """
        self._data = data

    # @abstractmethod
    # @property
    # def meta_data(self) -> GeneralMetaData:
    #     """
    #     Get the metadata information.
    #
    #     Returns:
    #         Info: The metadata associated with the data.
    #     """
    #     raise NotImplementedError

    # @property
    # def meta_handler(self) -> Union[CommonMetaHandler | None]:
    #     return None

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


class IDynamicData(IData):
    """Abstract base class for dynamic data."""

    def __init__(self, data: np.ndarray):
        super().__init__(data)

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
