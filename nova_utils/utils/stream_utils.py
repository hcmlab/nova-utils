"""Utility module for processing stream data
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 21.8.2023
"""

import math

def seconds_to_sample_nr(time_s: float, sr: float) -> float:
    """
    Calculates the specific sample number in a data stream that corresponds to a given time
    Args:
        sr (float): The sample rate of the data stream
        time_s (float): The timestamp of the sample in seconds

    Returns:
        float: Unrounded index of the sample in the stream for a given time

    """
    return time_s * sr


def milli_seconds_to_sample_nr(time_ms: int, sr: float) -> float:
    """
    Calculates the specific sample number in a data stream that corresponds to a given time
    Args:
        sr (float): The sample rate of the data stream
        time_ms (int): The timestamp of the sample in milliseconds

    Returns:
        float: Unrounded index of the sample in the stream for a given time

    """
    return seconds_to_sample_nr(sr=sr, time_s=time_ms / 1000.0)


def sample_nr_to_seconds(sample_nr: float, sr: float) -> float:
    """
    Calculates the specific time in seconds in a data stream that corresponds to given sample number
    Args:
        sample_nr (float): The unrounded sample number. Might represent a value between to actual samples of the stream.
        sr (float): The average numbers of samples per second.

    Returns:
        float: Unrounded time representation of sample_nr specified in seconds
    """
    return sample_nr * sr


def sample_nr_to_milli_seconds(sample_nr: float, sr: float) -> float:
    """
    Calculates the specific time in milli in a data stream that corresponds to given sample number
    Args:
        sample_nr (float): The unrounded sample number. Might represent a value between to actual samples of the stream.
        sr (float): The average numbers of samples per second.

     Returns:
         float: Unrounded time representation of sample_nr specified in milliseconds
    """
    return sample_nr_to_seconds(sample_nr, sr) * 1000.0


def time_to_sample_interval(
    start_time_ms: int, end_time_ms: int, sr: float
) -> tuple[int, int]:
    """
    Calculates the start and end sample number of a sliding window.
    Args:
        start_time_ms (int): Start time of the window in milliseconds
        end_time_ms (int): End time of the window in milliseconds
        sr (float): Sample rate of the data stream

    Returns:
        tuple[int, int]: First and last sample number of the window

    """

    # Get sample numbers
    start_sample_nr = milli_seconds_to_sample_nr(start_time_ms, sr)
    end_sample_nr = milli_seconds_to_sample_nr(end_time_ms, sr)

    # Rounding start and end to include maximum information
    start_sample_nr = math.floor(start_sample_nr)
    end_sample_nr = math.ceil(end_sample_nr)

    # Assert number of samples in window
    expected_number_of_samples = math.ceil(((end_time_ms - start_time_ms) / 1000) * sr)
    number_of_samples = end_sample_nr - start_sample_nr
    sample_diff = number_of_samples - expected_number_of_samples

    # Fewer then expected samples: We move start_sample number to the left
    # More samples than expected: We move start_sample number to the right
    start_sample_nr += sample_diff

    assert expected_number_of_samples == end_sample_nr - start_sample_nr

    return start_sample_nr, end_sample_nr
