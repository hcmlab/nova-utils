"""Definition of all Stream classes and Metadata
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 18.8.2023
"""

import numpy as np
from nova_utils.data.data import DynamicData
from nova_utils.utils.type_definitions import SSINPDataType
from nova_utils.utils.stream_utils import time_to_sample_interval


class StreamMetaData:
    """
    Metadata for a data stream, providing information about the stream properties.

    Attributes:
        duration (float): Duration of the stream in seconds.
        sample_shape (tuple): Shape of individual samples in the stream.
        num_samples (int): Total number of samples in the stream.
        sample_rate (float): Sampling rate of the stream in Hz.
        dtype (np.dtype): Data type of the samples.

    Args:
        duration (float, optional): Duration of the stream in seconds.
        sample_shape (tuple, optional): Shape of individual samples in the stream.
        num_samples (int, optional): Number of samples in the stream.
        sample_rate (float, optional): Sampling rate of the stream.
        dtype (np.dtype, optional): Data type of the samples.
    """

    def __init__(
        self,
        duration: float = None,
        sample_shape: tuple = None,
        num_samples: int = None,
        sample_rate: float = None,
        dtype: np.dtype = None,
    ):
        """
        Initialize a StreamMetaData instance with stream properties.
        """
        self.duration = duration
        self.sample_shape = sample_shape
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.dtype = dtype


class SSIStreamMetaData:
    """
    Metadata specific to SSI stream files.

    Attributes:
        chunks (np.ndarray): Chunks of the SSI stream with 'from', 'to', 'byte', and 'num' properties.

    Args:
        chunks (np.ndarray): Chunks of the SSI stream with 'from', 'to', 'byte', and 'num' properties.
    """

    CHUNK_DTYPE = np.dtype(
        [
            ("from", SSINPDataType.FLOAT.value),
            ("to", SSINPDataType.FLOAT.value),
            ("byte", SSINPDataType.INT.value),
            ("num", SSINPDataType.INT.value),
        ]
    )

    def __init__(self, chunks: np.ndarray):
        """
        Initialize an SSIStreamMetaData instance with chunks information.
        """
        self.chunks = chunks


class Stream(DynamicData):
    """
    A class representing a generic data stream along with associated metadata.

    This class extends the DynamicData class and implements methods for working
    with stream data.

    Attributes:
        (Inherits attributes from DynamicData.)

    Args:
        data (np.ndarray): The data stream.
        sample_rate (float): Sampling rate of the stream.
        duration (float, optional): Duration of the stream in seconds.Will be added to metadata.
        sample_shape (tuple, optional): Shape of individual samples in the stream. Will be added to metadata.
        num_samples (int, optional): Number of samples in the stream. Will be added to metadata.
        dtype (np.dtype, optional): Data type of the samples. Will be added to metadata.
        **kwargs: Additional keyword arguments for DynamicData.

    Methods:
        sample_from_interval(start: int, end: int) -> np.ndarray:
            Implementation of abstract method to sample data from within the specified interval.
    """

    def __init__(
        self,
        data: np.ndarray,
        sample_rate: float,
        duration: float = None,
        sample_shape: tuple = None,
        num_samples: int = None,
        dtype: np.dtype = None,
        **kwargs
    ):
        """
        Initialize a Stream instance with stream data and metadata.
        """
        super().__init__(data=data, **kwargs)

        # Add Metadata
        stream_meta_data = StreamMetaData(
            duration, sample_shape, num_samples, sample_rate, dtype
        )
        self.meta_data.expand(stream_meta_data)

    def sample_from_interval(self, start: int, end: int) -> np.ndarray:
        """
        Abstract method to sample data from within the specified interval.

        Args:
            start (int): The start index of the interval.
            end (int): The end index of the interval.

        Returns:
            np.ndarray: The sampled data within the interval.
        """

        start_sample, end_sample = time_to_sample_interval(start, end, self.meta_data.sample_rate)
        return self.data[start_sample : end_sample]


class SSIStream(Stream):
    """
    A class representing an SSI data stream.

    This class extends the Stream class with additional attributes specific to SSI streams.

    Attributes:
        (Inherits attributes from Stream.)
        CHUNK_DTYPE (np.dtype): Data type definition for SSI stream chunks.

    Args:
        data (np.ndarray): The SSI stream data.
        chunks (np.ndarray, optional): Chunks of the SSI stream.
        **kwargs: Additional keyword arguments for Stream.

    Methods:
        (No additional methods specified in the provided code.)
    """

    CHUNK_DTYPE = np.dtype(
        [
            ("from", SSINPDataType.FLOAT.value),
            ("to", SSINPDataType.FLOAT.value),
            ("byte", SSINPDataType.INT.value),
            ("num", SSINPDataType.INT.value),
        ]
    )

    def __init__(self, data: np.ndarray, chunks: np.ndarray = None, **kwargs):
        """
        Initialize an SSIStream instance with SSI stream data and metadata.
        """
        super().__init__(data=data, **kwargs)

        # Add Metadata
        ssistream_meta = SSIStreamMetaData(chunks=chunks)
        self.meta_data.expand(ssistream_meta)


class Audio(Stream):
    """
    A class representing an audio data stream.

    This class extends the Stream class with attributes and functionality specific to audio streams.

    Args:
        data (np.ndarray): The audio stream data.
        sample_rate (float): Sampling rate of the audio stream.
        **kwargs: Additional keyword arguments for Stream.

    Methods:
        (No additional methods specified in the provided code.)
    """

    def __init__(self, data: np.ndarray, sample_rate: float, **kwargs):
        super().__init__(data, sample_rate, **kwargs)


class Video(Stream):
    """
    A class representing video data stream.

    This class extends the Stream class with attributes and functionality specific to video streams.

    Args:
        data (np.ndarray): The video stream data.
        sample_rate (float): Sampling rate of the video stream.
        **kwargs: Additional keyword arguments for Stream.

    Methods:
        (No additional methods specified in the provided code.)
    """

    def __init__(self, data: np.ndarray, sample_rate: float, **kwargs):
        super().__init__(data, sample_rate, **kwargs)


if __name__ == "__main__":
    # Placeholder for main execution code
    ...
