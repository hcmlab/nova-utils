import numpy as np
import sys
from abc import ABC, abstractmethod
from numpy import dtype
from enum import Enum
from nova_utils.data.idata import IDynamicData, GeneralMetaData
from nova_utils.utils.anno_utils import get_overlap, get_anno_majority, is_garbage
import pandas as pd


# Schemes
class SchemeType(Enum):
    """Predefined annotation schemes"""

    DISCRETE = 0
    CONTINUOUS = 1
    FREE = 2

class LabelType(Enum):
    DISCRETE = np.dtype(
        [
            ("from", np.float64),
            ("to", np.float64),
            ("id", np.int32),
            ("conf", np.float32),
        ]
    )
    CONTINUOUS = np.dtype([("score", np.float32), ("conf", np.float32)])
    FREE = np.dtype(
        [
            ("from", np.float64),
            ("to", np.float64),
            ("name", np.object_),
            ("conf", np.float32),
        ]
    )


class IAnnotationScheme(ABC):
    def __init__(self, name: str):
        """
        Initialize the annotation scheme with the given name.

        Args:
            name (str): The name of the annotation scheme.
        """
        self.name = name

    @property
    @abstractmethod
    def scheme_type(self) -> SchemeType:
        """Get the type of the annotation scheme."""
        pass

    @property
    @abstractmethod
    def label_dtype(self) -> dtype:
        """Get the numpy data type of the labels used in the annotation scheme."""
        pass


class DiscreteAnnotationScheme(IAnnotationScheme):
    @property
    def label_dtype(self) -> dtype:
        return LabelType.DISCRETE.value

    @property
    def scheme_type(self) -> SchemeType:
        return SchemeType.DISCRETE

    def __init__(self, *args, classes: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = classes if classes else {}


class ContinuousAnnotationScheme(IAnnotationScheme):
    @property
    def label_dtype(self) -> dtype:
        return LabelType.CONTINUOUS.value

    @property
    def scheme_type(self) -> SchemeType:
        return SchemeType.CONTINUOUS

    def __init__(
        self, *args, sample_rate: float, min_val: float, max_val: float, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        self.min_val = min_val
        self.max_val = max_val


class FreeAnnotationScheme(IAnnotationScheme):
    @property
    def label_dtype(self) -> dtype:
        return LabelType.FREE.value

    @property
    def scheme_type(self) -> SchemeType:
        return SchemeType.FREE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Meta Information
class AnnoMetaData(GeneralMetaData):
    def __init__(self, *args, scheme: IAnnotationScheme, annotator: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotation_scheme = scheme
        self.annotator = annotator

# Annotations
class Annotation(IDynamicData):
    GARBAGE_LABEL_ID = np.NAN

    @property
    @abstractmethod
    def annotation_scheme(self) -> IAnnotationScheme:
        """Get the annotation scheme used for the data."""
        pass

    def __init__(
        self,
        data: np.ndarray,
        scheme : IAnnotationScheme,
    ):
        super().__init__(data=data)
        self._annotation_scheme = scheme

    @annotation_scheme.setter
    def annotation_scheme(self, value):
        pass

class DiscreteAnnotation(Annotation):

    # Class ids and string names as provided from NOVA-DB and required by SSI
    NOVA_REST_CLASS_NAME = "REST"
    NOVA_GARBAGE_LABEL_ID = -1

    # Initialize Rest class id with garbage class id
    REST_LABEL_ID = NOVA_GARBAGE_LABEL_ID

    def __init__(
            self,
            data: np.ndarray,
            scheme : DiscreteAnnotationScheme,
    ):
        super().__init__(data=data, scheme=scheme)
        self._data_values = None
        self._data_interval = None

    @property
    def annotation_scheme(self) -> DiscreteAnnotationScheme:
        assert isinstance(self._annotation_scheme, DiscreteAnnotationScheme)
        return self._annotation_scheme

    @annotation_scheme.setter
    def annotation_scheme(self, value):
        if not isinstance(value, DiscreteAnnotationScheme):
            raise TypeError(f"Expecting {DiscreteAnnotationScheme}, got {type(value)}.")
        self._annotation_scheme = value

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        assert value is None or value.dtype == self.annotation_scheme.label_dtype
        self._data = value
        if value is not None:
            df_tmp = pd.DataFrame(value)
            self._data_interval = df_tmp[["from", "to"]].values.astype(int)
            self._data_values = df_tmp[["id", "conf"]].values

    def sample_from_interval(self, start: int, end: int):

        overlap_idxs = get_overlap(self._data_interval, start, end)

        # If no label overlaps the requested frame we return rest class. If add_rest_class = False garbage label will be returned instead
        if not overlap_idxs.any():
            return self.REST_LABEL_ID

        majority_idx = get_anno_majority(self._data_interval, overlap_idxs, start, end)
        label = self._data_values[majority_idx, 0]
        if is_garbage(label, self.NOVA_GARBAGE_LABEL_ID):
            return self.GARBAGE_LABEL_ID
        return label

class FreeAnnotation(Annotation):
    """
    The FREE annotation scheme is used for any form of free text.
    """

    def __init__(
            self,
            data: np.ndarray,
            scheme : FreeAnnotationScheme,
    ):
        super().__init__(data=data, scheme=scheme)
        self._data_values = None
        self._data_interval = None

    @property
    def annotation_scheme(self) -> FreeAnnotationScheme:
        assert isinstance(self._annotation_scheme, FreeAnnotationScheme)
        return self._annotation_scheme

    @annotation_scheme.setter
    def annotation_scheme(self, value):
        if not isinstance(value, FreeAnnotationScheme):
            raise TypeError(f"Expecting {FreeAnnotationScheme}, got {type(value)}.")
        self._annotation_scheme = value

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value):
        assert value is None or value.dtype == self.annotation_scheme.label_dtype
        self._data = value
        if value is not None:
            df_tmp = pd.DataFrame(value)
            self._data_interval = df_tmp[["from", "to"]].values.astype(int)
            self._data_values = df_tmp[["name", "conf"]].values

    def sample_from_interval(self, start: int, end: int):
        annos_for_sample = get_overlap(self._data_interval, start, end)

        # No label matches
        if not annos_for_sample.any():
            return [""]

        return self._data_values[annos_for_sample, 0]

class ContinuousAnnotation(Annotation):

    # Class ids and string names as provided from NOVA-DB and required by SSI
    NOVA_GARBAGE_LABEL_VALUE = np.NAN
    MISSING_DATA_LABEL_VALUE = sys.float_info.min

    @property
    def annotation_scheme(self) -> ContinuousAnnotationScheme:
        assert isinstance(self._annotation_scheme, ContinuousAnnotationScheme)
        return self._annotation_scheme

    @annotation_scheme.setter
    def annotation_scheme(self, value):
        if not isinstance(value, ContinuousAnnotationScheme):
            raise TypeError(
                f"Expecting {ContinuousAnnotationScheme}, got {type(value)}."
            )
        self._annotation_scheme = value

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        assert value is None or value.dtype == self.annotation_scheme.label_dtype
        self._data = value

    def sample_from_interval(self, start, end):
        # returns zero if session duration is longer then labels
        s = int(start * self.annotation_scheme.sample_rate/ 1000)
        e = int(end * self.annotation_scheme.sample_rate / 1000)

        # Assure that indices for array are at least one integer apart
        if s == e:
            e = s + 1

        if len(self.data) >= e:
            frame = self.data[s:e]
            frame_data = frame[:, 0]
            frame_conf = frame[:, 1]
        else:
            return self.MISSING_DATA_LABEL_VALUE

        # TODO: Return timeseries instead of averagea
        conf = sum(frame_conf) / max(len(frame_conf), 1)
        label = sum(frame_data) / max(len(frame_data), 1)

        # If frame evaluates to garbage label discard sample
        if is_garbage(label, self.NOVA_GARBAGE_LABEL_VALUE):
            return self.NOVA_GARBAGE_LABEL_VALUE
        else:
            return label


if __name__ == "__main__":
    # Discrete anno
    discrete_scheme = DiscreteAnnotationScheme(
        name="disc_scheme", classes={0: "class_zero", 1: "class_one", 2: "class_two"}
    )
    discrete_data = np.array(
        [
            (0.5, 1.0, 0, 0.8),
            (1.5, 2.0, 2, 0.6),
            (2.5, 3.0, 1, 0.9),
        ],
        dtype=discrete_scheme.label_dtype,
    )

    discrete_anno = DiscreteAnnotation(
        data=discrete_data,
        scheme=discrete_scheme
    )

    # Continuous anno
    continuous_scheme = ContinuousAnnotationScheme(
        name="continuous_scheme", sample_rate=0.25, min_val=0, max_val=1
    )
    continuous_data = np.array(
        [
            (0.7292248, 0.52415526),
            (0.2252654, 0.4546865),
            (0.64103144, 0.7247994),
            (0.3928702, 0.5221592),
            (0.05887425, 0.58045745),
            (0.19909602, 0.01523399),
            (0.8669538, 0.8970701),
            (0.89999694, 0.80160624),
            (0.33919978, 0.7137072),
            (0.5318645, 0.53093654),
        ],
        dtype=continuous_scheme.label_dtype,
    )
    continuous_anno = ContinuousAnnotation(
        data=continuous_data,
        scheme=continuous_scheme,
    )

    # Free anno
    free_scheme = FreeAnnotationScheme(name="free_scheme")
    free_data = np.array(
        [
            (1.25, 2.75, "hello", 0.75),
            (3.14, 5.67, "world", 0.82),
            (0.25, 0.75, "yehaaaaw", 0.62),
            (7.89, 9.10, "!!!", 0.91),
        ],
        dtype=free_scheme.label_dtype,
    )
    free_anno = FreeAnnotation(
        data=free_data,
        scheme=free_scheme,
    )

    breakpoint()
