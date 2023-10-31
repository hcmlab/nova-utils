"""Utility module for all annotation data

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    21.8.2023

"""

from typing import Union

import numpy as np
from numba import njit

from nova_utils.utils.type_definitions import SSILabelDType, LabelDType
from nova_utils.utils.type_definitions import SchemeType


# TODO: Currently we do not take the rest class into account when calculating the label for the frame. Maybe we should do this
@njit
def get_overlap(a: np.ndarray, start: int, end: int):
    """
    Calculating all overlapping intervals between the given array of time intervals and the interval [start, end]

    Args:
        a (np.ndarray): numpy array of shape (n,2), where each entry contains an interval [from, to]
        start (int): start time of the interval to check in ms
        end (int): end time of the interval of the interval to check in ms

    Returns:
        Numpy array with boolean values. The array is true where the interval specified in a overlaps [start, end]

    """
    annos_for_sample = (
        # annotation is bigger than frame
        ((a[:, 0] <= start) & (a[:, 1] >= end))
        # end of annotation is in frame
        | ((a[:, 1] >= start) & (a[:, 1] <= end))
        # start of annotation is in frame
        | ((a[:, 0] >= start) & (a[:, 0] <= end))
    )
    return annos_for_sample


def get_anno_majority_distribution(a: np.ndarray, overlap_idxs: np.ndarray, start: int, end: int, num_classes: int) -> np.ndarray:
    """
    Returns an array of the distribution of annotations within the current frame

    Args:
        a (np.ndarray): numpy array of shape (1,2), where each entry contains an interval [from, to]
        overlap_idxs (np.ndarray): aray of boolean values where a is overlapping the interval [start, end] (as returned by get _get_overlap())
        start (int): start of the interval to check
        end (int): end of the interval to check
        num_classes (int): total number of classes

    Returns:
        np.ndarray numpy array containing the data distribution of the classes within the given frame. each index in the array matches the respective class id.
        np.NaN if a label is detected that ist negative or larger than num_classes
    """
    dist = np.zeros( (num_classes,) )

    # for each sample point where we have an overlap with the label
    for annotation in a[overlap_idxs]:
        dur = np.minimum(end, annotation['to']) - np.maximum(start, annotation['from'])
        if int(annotation['id']) not in range(num_classes):
            return np.NaN

        dist[annotation['id']] += dur

    # Rest class takes the rest amount of time
    dist[-1] = end-start-sum(dist)
    assert sum(dist) == end-start
    return dist / sum(dist)

def get_anno_majority(a: np.ndarray, overlap_idxs: np.ndarray, start: int, end: int):
    """
    Returns the index of the annotation with the largest overlap with the current frame

    Args:
        a (np.ndarray): numpy array of shape (1,2), where each entry contains an interval [from, to]
        overlap_idxs (np.ndarray): aray of boolean values where a is overlapping the interval [start, end] (as returned by get _get_overlap())
        start (int): start of the interval to check
        end (int): end of the interval to check

    Returns:

    """
    # TODO: rewrite for numba jit
    majority_index = -1
    overlap = 0
    for i in np.where(overlap_idxs)[0]:
        if (
            cur_overlap := np.minimum(end, a[i][1]) - np.maximum(start, a[i][0])
        ) > overlap:
            overlap = cur_overlap
            majority_index = i
    return majority_index


def label_is_garbage(label_id, garbage_label_id):
    """
    Check if a label is considered garbage.

    Args:
        label_id: The ID of the label to check.
        garbage_label_id: The ID of the garbage label.

    Returns:
        bool: True if the label is garbage, False otherwise.

    """
    # check for nan or compare with garbage label id
    if label_id != label_id or label_id == garbage_label_id:
        return True
    return False


def data_contains_garbage(data: Union[np.ndarray, int], garbage_label_id: object = np.NAN):
    """
    Check if a data array contains garbage values.

    Args:
        data (np.ndarray): The data array to check.
        garbage_label_id(object): The ID of the garbage label.

    Returns:
        bool: True if the data contains garbage values, False otherwise.

    """
    if isinstance(data, int):
        if data == garbage_label_id:
            return True
    # if data array is numerical
    if isinstance(data, float) or np.issubdtype(data.dtype, np.number):
            if np.isnan(data).any():
                return True
            elif (data != data).any():
                return True
            elif garbage_label_id in data:
                return True
            else:
                return False
    else:
        return any(np.vectorize(lambda x: isinstance(x, float) and np.isnan(x))(data))

def convert_label_to_ssi_dtype(
    data: np.ndarray, annotation_scheme_type: SchemeType
) -> np.ndarray:
    """
    Convert label data to SSILabelDType based on the annotation scheme type.

    Args:
        data (np.ndarray): The label data to convert.
        annotation_scheme_type (SchemeType): The annotation scheme type.

    Returns:
        np.ndarray: The converted label data with the appropriate SSILabelDType.

    """
    # Convert from milliseconds to seconds
    if annotation_scheme_type == SchemeType.DISCRETE:
        tmp_anno_data = data.astype(SSILabelDType.DISCRETE.value)
        tmp_anno_data["from"] = tmp_anno_data["from"] / 1000
        tmp_anno_data["to"] = tmp_anno_data["to"] / 1000
        return tmp_anno_data

    elif annotation_scheme_type == SchemeType.FREE:
        tmp_anno_data = data.astype(SSILabelDType.FREE.value)
        tmp_anno_data["from"] = tmp_anno_data["from"] / 1000
        tmp_anno_data["to"] = tmp_anno_data["to"] / 1000
        return tmp_anno_data
    elif annotation_scheme_type == SchemeType.CONTINUOUS:
        return data.astype(SSILabelDType.CONTINUOUS.value)
    else:
        raise ValueError(
            f"Annotation Scheme Type {annotation_scheme_type.name} mot supported"
        )


def convert_ssi_to_label_dtype(
    data: np.ndarray, annotation_scheme_type: SchemeType
) -> np.ndarray:
    """
    Convert SSILabelDType data to LabelDType based on the annotation scheme type.

    Args:
        data (np.ndarray): The SSILabelDType data to convert.
        annotation_scheme_type (SchemeType): The annotation scheme type.

    Returns:
        np.ndarray: The converted LabelDType data.

    """
    tmp_anno_data = data

    # Convert from milliseconds to seconds
    if annotation_scheme_type == SchemeType.DISCRETE:
        tmp_anno_data["from"] *= 1000
        tmp_anno_data["to"] *= 1000
        tmp_anno_data = data.astype(LabelDType.DISCRETE.value)
        return tmp_anno_data

    elif annotation_scheme_type == SchemeType.FREE:
        tmp_anno_data["from"] *= 1000
        tmp_anno_data["to"] *= 1000
        tmp_anno_data = data.astype(LabelDType.FREE.value)
        return tmp_anno_data

    elif annotation_scheme_type == SchemeType.CONTINUOUS:
        return tmp_anno_data.astype(LabelDType.CONTINUOUS.value)

    else:
        raise ValueError(
            f"Annotation Scheme Type {annotation_scheme_type.name} mot supported"
        )


def _pack(data : np.ndarray[LabelDType.DISCRETE], max_time_gap=0):

    # Conditions to stop label aggregation
    label_changes = data['id'][:-1] != data['id'][1:]
    larger_than_max_gap = data['from'][1:] - data['to'][:-1] > max_time_gap
    change = [a or b for a,b in zip(label_changes, larger_than_max_gap)]

    split_data = np.split(data, np.where(change)[0]+1)

    # Aggregate all data clusters to one new label
    agg_data = np.asarray([ (x[0]['from'], x[-1]['to'], x[-1]['id'], np.mean(x['conf']) ) for x in split_data], dtype=data.dtype)
    return agg_data


def _remove(data: np.ndarray[LabelDType.DISCRETE], min_dur: int = 0):
    return np.delete(data, np.where(data['to'] - data['from'] < min_dur)[0])

def pack_remove(data : np.ndarray[LabelDType.DISCRETE], min_gap: int = 0, min_dur: int = 0) -> np.ndarray:
    '''
    Aggregate consecutive annotations with the same label.
    Does only work with discrete label data.

    Args:
        max_time_gap (int): The minimum amount of time between consecutive samples to be seen as two different samples. Defaults to 0.
        min_dur (int): Minimum duration of one sample Defaults to 0.

    Returns:

    '''

    data_copy = data.copy()
    try:
        # Pack
        data_copy = _pack(data_copy, min_gap)

        # Remove
        data_copy = _remove(data_copy, min_dur)

        # Pack
        data_copy = _pack(data_copy, min_gap)

    except Exception as e:
        print(f'Exception during data packing. Returning empty Annotation: {e}')
        return np.array([])

    return data_copy
