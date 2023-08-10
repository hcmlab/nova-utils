from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

class DataType(Enum):
    """Enumeration of possible data types."""

    # Dynamic signals
    SIGNAL_VIDEO = 0
    SIGNAL_AUDIO = 1
    SIGNAL_FEATURE = 2

    # Static signals
    SIGNAL_IMAGE = 3

    # Annotations
    ANNOTATION_DISCRETE = 4
    ANNOTATION_CONTINUOUS = 5
    ANNOTATION_FREE = 6
    ANNOTATION_POINT = 7
    ANNOTATION_DISCRETE_POLYGON = 8


class MetaData:
    """Container for metadata information."""
    # TODO define interfaces for separate metadata objects
    def __init__(self, general: object = None, signal: object = None, handler: object = None):
        """
        Initialize metadata.

        Args:
            general (object, optional): General metadata for the data. E.g. country, dataset
            signal (object, optional): Metadata that is dependent on the specific data type. E.g. codec, duration
            handler (object, optional): Metadata that is dependent on the specific data handler. E.g. filepath, database connection
        """
        self.general = general
        self.signal = signal
        self.handler = handler


class IData(ABC):
    """Abstract base class for all data types."""

    def __init__(self, data: np.ndarray = None, meta_data: MetaData = None):
        """
        Initialize data.

        Args:
            data (np.ndarray, optional): The data array. Defaults to None.
            meta_data (MetaInfo, optional): Metadata information. Defaults to None.
        """
        self._data = data
        self._meta_data = meta_data
        self._lazy_loading = False

    @property
    def meta_data(self) -> MetaData:
        """
        Get the metadata information.

        Returns:
            MetaData: The metadata associated with the data.
        """
        return self._meta_data

    @meta_data.setter
    def meta_data(self, value):
        self._meta_data = value

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

    # @abstractmethod
    # def _eager_sample_from_interval(self, start: int, end: int) -> np.ndarray:
    #     """
    #     Sample data from the specified interval.
    #
    #     Args:
    #         start (int): The start time of the interval in milliseconds.
    #         end (int): The end time of the interval in milliseconds.
    #
    #     Returns:
    #         np.ndarray: The sampled data within the specified interval.
    #     """
    #     pass
    #
    # @abstractmethod
    # def _lazy_sample_from_interval(self, start: int, end: int) -> np.ndarray:
    #     """
    #     Sample data from the specified time interval.
    #
    #     Args:
    #         start (int): The start time of the interval in milliseconds.
    #         end (int): The end time of the interval in milliseconds.
    #
    #     Returns:
    #         np.ndarray: The sampled data within the specified interval.
    #     """
    #     pass

    def sample_from_interval(self, start: int, end: int) -> np.ndarray:
        """
        Sample data from the specified time interval.

        Args:
            start (int): The start time of the interval in milliseconds.
            end (int): The end time of the interval in milliseconds.

        Returns:
            np.ndarray: The sampled data within the specified interval.
        """
        if self._lazy_loading:
            return self._lazy_sample_from_interval(start, end)
        else:
            if self.data is None:

                return self._eager_sample_from_interval(start, end)


class ITimeDiscreteData(IDynamicData, ABC):
    """Abstract base class for time-discrete dynamic data. Placeholder for future usage."""


class ITimeContinuousData(IDynamicData, ABC):
    """Abstract base class for time-continuous dynamic data. Placeholder for future usage."""


    def __init__(self, *args, sample_rate=None, **kwargs):
        """
        Initialize time-continuous dynamic data.

        Args:
            sample_rate (float, optional): The sample rate of the data. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self._sample_rate = sample_rate


    @property
    def sample_rate(self) -> float:
        """
        Get the sample rate of the data.

        Returns:
            float: The sample rate.
        """
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value


class IValueContinuousData(IData):
    """Abstract base class for continuous value data. Placeholder for future usage."""


class IValueDiscreteData(IData):
    """Abstract base class for discrete value data. Placeholder for future usage."""
