"""Utility module for all annotation data
Author: Dominik Schiller
Date: 21.8.2023
"""

import numpy as np
from numba import njit
from nova_utils.utils.type_definitions import SSILabelDType, LabelDType
from nova_utils.utils.type_definitions import SchemeType
from typing import Union

# TODO: Currently we do not take the rest class into account when calculating the label for the frame. Maybe we should do this
@njit
def get_overlap(a: np.ndarray, start: int, end: int):
    """
    Calculating all overlapping intervals between the given array of time intervals and the interval [start, end]
    Args:
        a (): numpy array of shape (n,2), where each entry contains an interval [from, to]
        start (): start time of the interval to check in ms
        end (): end time of the interval of the interval to check in ms

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


def get_anno_majority(a, overlap_idxs: np.ndarray, start: int, end: int):
    """
    Returns the index of the annotation with the largest overlap with the current frame
    Args:
        a (): numpy array of shape (1,2), where each entry contains an interval [from, to]
        overlap_idxs (): aray of boolean values where a is overlapping the interval [start, end] (as returned by get _get_overlap())
        start (): start of the interval to check
        end (): end of the interval to check

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


def is_garbage(local_label_id, nova_garbage_label_id):
    # check for nan or compare with garbage label id
    if local_label_id != local_label_id or local_label_id == nova_garbage_label_id:
        return True
    return False

def convert_label_to_ssi_dtype(data: np.ndarray, annotation_scheme_type: SchemeType) -> np.ndarray:

    # Convert from milliseconds to seconds
    if annotation_scheme_type == SchemeType.DISCRETE:
        tmp_anno_data = data.astype(SSILabelDType.DISCRETE.value)
        tmp_anno_data['from'] = tmp_anno_data['from'] / 1000
        tmp_anno_data['to'] = tmp_anno_data['to'] / 1000
        return tmp_anno_data

    elif annotation_scheme_type == SchemeType.FREE:
        tmp_anno_data = data.astype(SSILabelDType.FREE.value)
        tmp_anno_data['from'] = tmp_anno_data['from'] / 1000
        tmp_anno_data['to'] = tmp_anno_data['to'] / 1000
        return tmp_anno_data
    elif annotation_scheme_type == SchemeType.CONTINUOUS:
        return data.astype(SSILabelDType.CONTINUOUS.value)
    else:
        raise ValueError(f'Annotation Scheme Type {annotation_scheme_type.name} mot supported')


def convert_ssi_to_label_dtype(data: np.ndarray, annotation_scheme_type: SchemeType) -> np.ndarray:

    tmp_anno_data = data

    # Convert from milliseconds to seconds
    if annotation_scheme_type == SchemeType.DISCRETE:
        tmp_anno_data['from'] *= 1000
        tmp_anno_data['to'] *=  1000
        tmp_anno_data = data.astype(LabelDType.DISCRETE.value)
        return tmp_anno_data

    elif annotation_scheme_type == SchemeType.FREE:
        tmp_anno_data['from'] *= 1000
        tmp_anno_data['to'] *=  1000
        tmp_anno_data = data.astype(LabelDType.FREE.value)
        return tmp_anno_data

    elif annotation_scheme_type == SchemeType.CONTINUOUS:
        return tmp_anno_data.astype(LabelDType.CONTINUOUS.value)

    else:
        raise ValueError(f'Annotation Scheme Type {annotation_scheme_type.name} mot supported')

