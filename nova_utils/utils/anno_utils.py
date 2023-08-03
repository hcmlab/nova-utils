import numpy as np
from numba import njit

# TODO: Currently we do not take the rest class into account when calculating the label for the frame. Maybe we should do this
@njit
def get_overlap(a, start, end):
    """
    Calculating all overlapping intervals between the given array of time intervals and the interval [start, end]
    Args:
        a (): numpy array of shape (1,2), where each entry contains an interval [from, to]
        start (): start of the interval to check
        end (): end of the interval to check

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


def get_anno_majority(a, overlap_idxs, start, end):
    """
    Returns the index of the annotation with the largest overlap with the current frame
    Args:
        a (): numpy array of shape (1,2), where each entry contains an interval [from, to]
        overlap_idxs (): aray of boolean values where a is overlapping the intervall [start, end] (as returned by get _get_overlap())
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