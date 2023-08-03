from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class DataType(Enum):

    # Dynamic signals
    SIGNAL_VIDEO = 0
    SIGNAL_AUDIO  = 1
    SIGNAL_FEATURE = 2

    # Static signals
    SIGNAL_IMAGE = 3

    # Annotations
    ANNOTATION_DISCRETE = 4
    ANNOTATION_CONTINUOUS = 5
    ANNOTATION_FREE = 6
    ANNOTATION_POINT = 7
    ANNOTATION_DISCRETE_POLYGON = 8

class MetaInfo():
    def __init__(self, role:str = None, dataset:str = None, session:str = None):
        self.role = role
        self.dataset = dataset
        self.session = session

class IData(ABC):
    ''' Abstract base class for all data types '''

    def __init__(self, data: np.ndarray = None, meta_info : MetaInfo = None):
        self._data = data
        self._meta_info = meta_info

    @property
    def meta_info(self) -> MetaInfo:
        return self._meta_info

    @meta_info.setter
    def meta_info(self, value):
        self._meta_info = value

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value):
        self._data = value


class IStaticData(IData):
    pass

class IDynamicData(IData):
    @abstractmethod
    def sample_from_interval(self, start: int, end: int) -> np.ndarray:
        pass

class ITimeDiscreteData(IDynamicData):
    pass

class ITimeContinuousData(IDynamicData):

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        return self._sample_rate

    def __init__(self, *args, sample_rate = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_rate = sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value


class IValueContinuousData(IData):
    pass

class IValueDiscreteData(IData):
    pass