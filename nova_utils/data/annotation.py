"""Definition of all Annotation classes and Metadata
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 18.8.2023
"""
import sys
from abc import ABC, abstractmethod

import numpy as np
from numpy import dtype

from nova_utils.data.data import DynamicData
from nova_utils.utils.anno_utils import get_overlap, label_is_garbage
from nova_utils.utils.type_definitions import LabelDType, SchemeType


class AnnoMetaData:
    """
    Metadata for annotations, providing information about the annotator.

    Attributes:
        annotator (str, optional): Annotator identifier.
        duration (float, optional): Duration of the stream in ms.

    Args:
        annotator (str, optional): Annotator identifier.
    """

    def __init__(self, annotator: str = None, duration: int = None):
        """
        Initialize an AnnoMetaData instance with annotator information.
        """
        self.annotator = annotator
        self.duration = duration


class IAnnotationScheme(ABC):
    """
    Abstract base class for annotation schemes.

    This class defines the interface for annotation schemes used in the system.

    Attributes:
        name (str): The name of the annotation scheme.

    Args:
        name (str): The name of the annotation scheme.
    """

    def __init__(self, name: str):
        """
        Initialize an IAnnotationScheme instance with the given name.
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
    """
    Discrete annotation scheme class.

    Attributes:
        classes (dict): Dictionary mapping class IDs to class names.

    Args:
        classes (dict): Dictionary mapping class IDs to class names.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, classes: dict, **kwargs):
        """
        Initialize a DiscreteAnnotationScheme instance with the provided class information.
        """
        super().__init__(*args, **kwargs)
        self.classes = classes

    @classmethod
    @property
    def label_dtype(self) -> dtype:
        """Get the numpy data type of discrete labels."""
        return LabelDType.DISCRETE.value

    @classmethod
    @property
    def scheme_type(self) -> SchemeType:
        """Get the type of the annotation scheme (Discrete)."""
        return SchemeType.DISCRETE


class ContinuousAnnotationScheme(IAnnotationScheme):
    """
    Continuous annotation scheme class.

    Attributes:
        sample_rate (float): Sampling rate of the continuous data.
        min_val (float): Minimum value of the continuous data.
        max_val (float): Maximum value of the continuous data.

    Args:
        sample_rate (float): Sampling rate of the continuous data.
        min_val (float): Minimum value of the continuous data.
        max_val (float): Maximum value of the continuous data.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self, *args, sample_rate: float, min_val: float, max_val: float, **kwargs
    ):
        """
        Initialize a ContinuousAnnotationScheme instance with the provided sampling rate and value range.
        """
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        self.min_val = min_val
        self.max_val = max_val

    @classmethod
    @property
    def label_dtype(self) -> dtype:
        """Get the numpy data type of continuous labels."""
        return LabelDType.CONTINUOUS.value

    @classmethod
    @property
    def scheme_type(self) -> SchemeType:
        """Get the type of the annotation scheme (Continuous)."""
        return SchemeType.CONTINUOUS


class FreeAnnotationScheme(IAnnotationScheme):
    """
    Free annotation scheme class.

    This scheme is used for any form of free text.

    Args:
        name (str): The name of the annotation scheme.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a FreeAnnotationScheme instance.
        """
        super().__init__(*args, **kwargs)

    @classmethod
    @property
    def label_dtype(self) -> dtype:
        """Get the numpy data type of free text labels."""
        return LabelDType.FREE.value

    @classmethod
    @property
    def scheme_type(self) -> SchemeType:
        """Get the type of the annotation scheme (Free)."""
        return SchemeType.FREE


class Annotation(DynamicData):
    """
    Base class for annotations.

    Attributes:
        GARBAGE_LABEL_ID (float): Constant representing garbage label.

    Args:
        data (np.ndarray, optional): Array of annotation data.
        scheme (IAnnotationScheme): The annotation scheme used for the data.
        annotator (str, optional): Annotator identifier. Will be added to metadata.
        duration (float, optional): Duration of the annotation in seconds.Will be added to metadata.
        **kwargs: Arbitrary keyword arguments.
    """

    GARBAGE_LABEL_ID = np.NAN

    def __init__(
        self,
        scheme: IAnnotationScheme,
        annotator: str = None,
        duration: float = None,
        **kwargs,
    ):
        """
        Initialize an Annotation instance with data, scheme, and annotator information.
        """
        super().__init__(**kwargs)
        self.annotation_scheme = scheme

        # Create meta data
        anno_meta_data = AnnoMetaData(annotator=annotator, duration=duration)
        self.meta_data.expand(anno_meta_data)

    @property
    @abstractmethod
    def annotation_scheme(self) -> IAnnotationScheme:
        """Get the annotation scheme used for the data."""
        pass

    @annotation_scheme.setter
    def annotation_scheme(self, value):
        pass


class DiscreteAnnotation(Annotation):
    """
    Discrete annotation class.

    Attributes:
        NOVA_REST_CLASS_NAME (int): Constant representing NOVA rest class Name.
        NOVA_GARBAGE_LABEL_ID (int): Constant representing NOVA garbage label.
        rest_label_id (int): Class-variable representing the nova rest class id.

    Args:
        data (np.ndarray): Array of discrete annotation data.
        scheme (DiscreteAnnotationScheme): The discrete annotation scheme used for the data.
        **kwargs: Arbitrary keyword arguments.
    """

    # Class ids and string names as provided from NOVA-DB and required by SSI
    NOVA_REST_CLASS_NAME = "REST"
    NOVA_GARBAGE_LABEL_ID = -1

    def __init__(
        self, scheme: DiscreteAnnotationScheme, data: np.ndarray = None, **kwargs
    ):
        """
        Initialize a DiscreteAnnotation instance with data and scheme information.
        """
        super().__init__(scheme=scheme, **kwargs)
        self._data_values = None
        self._data_interval = None
        self.data = data

    @property
    def annotation_scheme(self) -> DiscreteAnnotationScheme:
        """Get the discrete annotation scheme used for the data."""
        assert isinstance(self._annotation_scheme, DiscreteAnnotationScheme)
        return self._annotation_scheme

    @annotation_scheme.setter
    def annotation_scheme(self, value):
        if not isinstance(value, DiscreteAnnotationScheme):
            raise TypeError(f"Expecting {DiscreteAnnotationScheme}, got {type(value)}.")
        self._annotation_scheme = value
        # Rest class ID is last label ID + 1
        self.rest_label_id = len(self._annotation_scheme.classes)

    @property
    def data(self) -> np.ndarray:
        """Get the annotation data."""
        return self._data

    @data.setter
    def data(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=self.annotation_scheme.label_dtype)
        assert value is None or value.dtype == self.annotation_scheme.label_dtype
        self._data = value
        if value is not None:
            # Extract 'from' and 'to' columns and create a new array
            self._data_interval = np.empty((len(value), 2), dtype="<i4")
            self._data_interval[:, 0] = value["from"]
            self._data_interval[:, 1] = value["to"]
            # df_tmp = pd.DataFrame(value)
            # self._data_interval = value[['from','to']]
            # self._data_values = df_tmp[["id", "conf"]].values

    def sample_from_interval(self, start: int, end: int):
        # TODO handle self.data == None
        """
        Sample annotation from an interval.

        Args:
            start (int): Start of the interval in milliseconds.
            end (int): End of the interval milliseconds.

        Returns:
            int: Sampled annotation label.
        """

        overlap_idxs = get_overlap(self._data_interval, start, end)
        dist = np.zeros(
            (len(self.annotation_scheme.classes) + 1),
        )

        # For each sample point where we have an overlap with the label
        for annotation in self.data[overlap_idxs]:
            dur = np.minimum(end, annotation["to"]) - np.maximum(
                start, annotation["from"]
            )

            # If one label is garbage return garbage
            if int(annotation["id"]) is self.NOVA_GARBAGE_LABEL_ID:
                return self.GARBAGE_LABEL_ID

            dist[annotation["id"]] += dur

        # Rest class takes the rest amount of time
        dist[-1] = end - start - sum(dist)

        return dist / sum(dist)


class FreeAnnotation(Annotation):
    """
    Free annotation class.

    Attributes:
        NOVA_GARBAGE_LABEL_VALUE (float): Constant representing NOVA garbage label value.

    Args:
        data (np.ndarray): Array of free text annotation data.
        scheme (FreeAnnotationScheme): The free annotation scheme used for the data.
    """

    NOVA_GARBAGE_LABEL_VALUE = np.NAN

    def __init__(self, scheme: FreeAnnotationScheme, data: np.ndarray = None, **kwargs):
        """
        Initialize a FreeAnnotation instance with data and scheme information.
        """
        super().__init__(scheme=scheme, **kwargs)
        self._data_values = None
        self._data_interval = None
        self.data = data

    @property
    def annotation_scheme(self) -> FreeAnnotationScheme:
        """Get the free annotation scheme used for the data."""
        assert isinstance(self._annotation_scheme, FreeAnnotationScheme)
        return self._annotation_scheme

    @annotation_scheme.setter
    def annotation_scheme(self, value):
        if not isinstance(value, FreeAnnotationScheme):
            raise TypeError(f"Expecting {FreeAnnotationScheme}, got {type(value)}.")
        self._annotation_scheme = value

    @property
    def data(self) -> np.ndarray:
        """Get the annotation data."""
        return self._data

    @data.setter
    def data(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=self.annotation_scheme.label_dtype)
        assert value is None or value.dtype == self.annotation_scheme.label_dtype
        self._data = value
        if value is not None:
            # Extract 'from' and 'to' columns and create a new array
            self._data_interval = np.empty((len(value), 2), dtype="<i4")
            self._data_interval[:, 0] = value["from"]
            self._data_interval[:, 1] = value["to"]

            # self._data_interval = value[['from', 'to']]
            # self._data_values = value[['name', 'conf']]
            # df_tmp = pd.DataFrame(value)
            # self._data_interval = df_tmp[["from", "to"]].values
            # self._data_values = df_tmp[["name", "conf"]].values

    def sample_from_interval(self, start: int, end: int):
        """
        Sample annotation from an interval.

        Args:
            start (int): Start of the interval in milliseconds.
            end (int): End of the interval in milliseconds.

        Returns:
            np.ndarray: Numpy array of sampled annotation labels. If no annotations are falling in the specified interval an empty numpy array will be returned.
        """
        annos_for_sample = get_overlap(self._data_interval, start, end)

        if not annos_for_sample.any():
            return np.asarray([])

        return self.data[annos_for_sample]["name"]

    # return self._data_values[annos_for_sample, 0]


class ContinuousAnnotation(Annotation):
    """
    Continuous annotation class.

    Attributes:
        NOVA_GARBAGE_LABEL_VALUE (float): Constant representing NOVA garbage label value.
        MISSING_DATA_LABEL_VALUE (float): Constant representing missing data label value.

    Args:
        data (np.ndarray): Array of continuous annotation data.
        scheme (ContinuousAnnotationScheme): The continuous annotation scheme used for the data.
        **kwargs: Arbitrary keyword arguments.
    """

    NOVA_GARBAGE_LABEL_VALUE = np.NAN
    MISSING_DATA_LABEL_VALUE = sys.float_info.min

    def __init__(
        self, scheme: ContinuousAnnotationScheme, data: np.ndarray = None, **kwargs
    ):
        """
        Initialize a DiscreteAnnotation instance with data and scheme information.
        """
        super().__init__(scheme=scheme, **kwargs)
        self.data = data

    @property
    def annotation_scheme(self) -> ContinuousAnnotationScheme:
        """Get the continuous annotation scheme used for the data."""
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
        """Get the annotation data."""
        return self._data

    @data.setter
    def data(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            value = np.asarray(value, dtype=self.annotation_scheme.label_dtype)
        assert value is None or value.dtype == self.annotation_scheme.label_dtype
        self._data = value

    def sample_from_interval(self, start, end):
        """
        Sample annotation from an interval.

        Args:
            start (int): Start of the interval in milliseconds.
            end (int): End of the interval in milliseconds.

        Returns:
            float: Sampled annotation label.
        """
        s = int(start * self.annotation_scheme.sample_rate / 1000)
        e = int(end * self.annotation_scheme.sample_rate / 1000)

        if s == e:
            e = s + 1

        if len(self.data) >= e:
            frame = self.data[s:e]
            frame_data = frame[:, 0]
            frame_conf = frame[:, 1]
        else:
            return self.MISSING_DATA_LABEL_VALUE

        # conf = sum(frame_conf) / max(len(frame_conf), 1)
        label = sum(frame_data) / max(len(frame_data), 1)

        if label_is_garbage(label, self.NOVA_GARBAGE_LABEL_VALUE):
            return self.NOVA_GARBAGE_LABEL_VALUE
        else:
            return label


if __name__ == "__main__":
    # Discrete annotation example
    discrete_scheme = DiscreteAnnotationScheme(
        name="disc_scheme", classes={0: "class_zero", 1: "class_one", 2: "class_two"}
    )
    discrete_data = np.array(
        [
            (500, 1000, 0, 0.8),
            (1500, 2000, 2, 0.6),
            (2500, 3000, 1, 0.9),
        ],
        dtype=discrete_scheme.label_dtype,
    )

    discrete_anno = DiscreteAnnotation(data=discrete_data, scheme=discrete_scheme)

    # Continuous annotation example
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

    # Free annotation example
    free_scheme = FreeAnnotationScheme(name="free_scheme")
    free_data = np.array(
        [
            (1250, 2750, "hello", 0.75),
            (3140, 5670, "world", 0.82),
            (250, 750, "yehaaaaw", 0.62),
            (7890, 9100, "!!!", 0.91),
        ],
        dtype=free_scheme.label_dtype,
    )
    free_anno = FreeAnnotation(
        data=free_data,
        scheme=free_scheme,
    )
