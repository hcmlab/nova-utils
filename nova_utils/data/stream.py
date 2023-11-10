"""Definition of all Stream classes and Metadata
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 18.8.2023
"""

from typing import Union

import numpy as np

from nova_utils.data.data import DynamicData
from nova_utils.utils.stream_utils import time_to_sample_interval
from nova_utils.utils.type_definitions import SSINPDataType


class StreamMetaData:
    """
    Metadata for a data stream, providing information about the stream properties.

    Attributes:
        ext (str): File extension of the stream including the leading '.'
        duration (float): Duration of the stream in seconds.
        sample_shape (tuple): Shape of individual samples in the stream.
        num_samples (int): Total number of samples in the stream.
        sample_rate (float): Sampling rate of the stream in Hz.
        dtype (np.dtype): Data type of the samples.
        media_type (string, optional): Media type of the stream data as specified in NOVA-DB. Defaults to feature.
        custom_meta (dict, optional): Stream type specific meta information to add. E.g. aspect ratio of processed video.


    Args:
        ext (str): File extension of the stream including the leading '.'
        duration (float, optional): Duration of the stream in seconds.
        sample_shape (tuple, optional): Shape of individual samples in the stream.
        num_samples (int, optional): Number of samples in the stream.
        sample_rate (float, optional): Sampling rate of the stream.
        dtype (np.dtype, optional): Data type of the samples.
        media_type (string, optional): Media type of the stream data as specified in NOVA-DB. Defaults to feature.
        custom_meta (dict, optional): Stream type specific meta information to add. E.g. aspect ratio of processed video.
    """

    def __init__(
        self,
        ext: str = None,
        duration: float = None,
        sample_shape: tuple = None,
        num_samples: int = None,
        sample_rate: float = None,
        dtype: np.dtype = None,
        media_type: str = 'feature',
        custom_meta: dict = None
    ):
        """
        Initialize a StreamMetaData instance with stream properties.
        """
        self.ext = ext
        self.duration = duration
        self.sample_shape = sample_shape
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.media_type = media_type
        self.custom_meta = custom_meta if custom_meta is not None else {}


class SSIStreamMetaData:
    """
    Metadata specific to SSI stream files.

    Attributes:
        chunks (np.ndarray): Chunks of the SSI stream with 'from', 'to', 'byte', and 'num' properties.
        dimlabels (list[dict]): List of dictionaries mapping an integer id as key (stream index) to a descriptive string as value.

    Args:
        chunks (np.ndarray): Chunks of the SSI stream with 'from', 'to', 'byte', and 'num' properties.
        dimlabels (list[dict]): List of dictionaries mapping an integer id as key (stream index) to a descriptive string as value.

    """

    CHUNK_DTYPE = np.dtype(
        [
            ("from", SSINPDataType.FLOAT.value),
            ("to", SSINPDataType.FLOAT.value),
            ("byte", SSINPDataType.INT.value),
            ("num", SSINPDataType.INT.value),
        ]
    )

    def __init__(self, chunks: np.ndarray, dim_labels: list = None):
        """
        Initialize an SSIStreamMetaData instance with chunks information.
        """
        self.chunks = chunks
        self.dim_labels = dim_labels


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
        dtype (np.dtype, optional): Data type of the samples. Will be added to metadata. Defaults to np.float32 .
        name (str, optional): Name of the stream.
        ext (str, optional): File extension of the stream including the leading '.'. Defaults to '.stream', '.mp4' or '.wav' depending on the stream type.
        media_type (string, optional): Media type of the stream data as specified in NOVA-DB. Defaults to feature.
        custom_meta (dict, optional): Stream type specific meta information to add. E.g. aspect ratio of processed video.
        **kwargs: Additional keyword arguments for DynamicData.

    """

    def __init__(
        self,
        data: Union[np.ndarray, None],
        sample_rate: float,
        ext: str = None,
        duration: float = None,
        sample_shape: tuple = None,
        num_samples: int = None,
        dtype: np.dtype = SSINPDataType.FLOAT.value,
        media_type: str = '',
        custom_meta: dict = None,
        **kwargs
    ):
        """
        Initialize a Stream instance with stream data and metadata.
        """
        super().__init__(data=data, **kwargs)

        # Add Metadata
        if ext is None:
            if isinstance(self, SSIStream):
                ext = '.stream'
            if isinstance(self, Video):
                ext = '.mp4'
            if isinstance(self, Audio):
                ext = '.wav'

        stream_meta_data = StreamMetaData(
            ext, duration, sample_shape, num_samples, sample_rate, dtype, media_type, custom_meta
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
        data (np.ndarray): The SSI stream data. Shape is (num_samples,) + (sample_shape,)
        sample_rate (float): Sampling rate of the SSI stream.
        chunks (np.ndarray, optional): Chunks of the SSI stream.
        **kwargs: Additional keyword arguments for Stream.

    Methods:
        (No additional methods specified in the provided code.)
    """

    CHUNK_DTYPE = np.dtype(
        [
            ("from", SSINPDataType.FLOAT.value), # start of chunk in seconds
            ("to", SSINPDataType.FLOAT.value), # end of chunk in seconds
            ("byte", SSINPDataType.INT.value), # number of bytes for the chunk
            ("num", SSINPDataType.INT.value), # number of samples for the chunk
        ]
    )

    def __init__(self, data: np.ndarray, sample_rate: float, chunks: np.ndarray = None, dim_labels: list = None, **kwargs):
        """
        Initialize an SSIStream instance with SSI stream data and metadata.
        """
        super().__init__(data=data, sample_rate=sample_rate, **kwargs)

        # Add Metadata
        if data is not None and chunks is None:
            num_samples = data.shape[0]
            chunks = np.asarray([(0, num_samples / sample_rate, 0, num_samples)], dtype=self.CHUNK_DTYPE)
        ssistream_meta = SSIStreamMetaData(chunks=chunks, dim_labels=dim_labels)
        self.meta_data.expand(ssistream_meta)


class Audio(Stream):
    """
    A class representing an audio data stream.

    This class extends the Stream class with attributes and functionality specific to audio streams.

    Args:
        data (np.ndarray): The audio stream data. Shape is (num_channels, num_samples). Dtype is float.
        sample_rate (float): Sampling rate of the audio stream.
        **kwargs: Additional keyword arguments for Stream.

    """

    def __init__(self, data: np.ndarray, sample_rate: float, **kwargs):
        super().__init__(data=data, sample_rate=sample_rate, **kwargs)


class Video(Stream):
    """
    A class representing video data stream.

    This class extends the Stream class with attributes and functionality specific to video streams.

    Args:
        data (np.ndarray): The video stream data. Shape is (num_samples, height, width, num_channels)
        sample_rate (float): Sampling rate of the video stream.
        **kwargs: Additional keyword arguments for Stream.

    """

    def __init__(self, data: np.ndarray, sample_rate: float, **kwargs):
        super().__init__(data=data, sample_rate=sample_rate, **kwargs)


if __name__ == "__main__":
    # Placeholder for main execution code
    ...
