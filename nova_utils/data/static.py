"""Definition of all Stream classes and Metadata
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 18.8.2023
"""

import numpy as np

from nova_utils.data.data import StaticData


class StaticMetaData:
    """
    Metadata for a data stream, providing information about the stream properties.

    Attributes:
        name (str): Name of the stream.
        ext (str): File extension of the stream including the leading '.'
        duration (float): Duration of the stream in seconds.
        sample_shape (tuple): Shape of individual samples in the stream.
        num_samples (int): Total number of samples in the stream.
        sample_rate (float): Sampling rate of the stream in Hz.
        dtype (np.dtype): Data type of the samples.
        media_type (string, optional): Media type of the stream data as specified in NOVA-DB. Defaults to feature.
        custom_meta (dict, optional): Stream type specific meta information to add. E.g. aspect ratio of processed video.


    Args:
        name (str): Name of the stream.
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
            name: str = None,
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
        self.name = name
        self.ext = ext
        self.duration = duration
        self.sample_shape = sample_shape
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.dtype = dtype
        self.media_type = media_type
        self.custom_meta = custom_meta if custom_meta is not None else {}

class Image(StaticData):
    """
    A class representing static image.

    This class extends the Stream class with attributes and functionality specific to single images.

    Args:
        data (np.ndarray): The image data. Shape is (height, width, num_channels)
        **kwargs: Additional keyword arguments for Stream.

    """

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data=data, **kwargs)

class Text(StaticData):
    """
    A class representing text.

    This class extends the Stream class with attributes and functionality specific to text.

    Args:
        data (np.ndarray): The text data.
        **kwargs: Additional keyword arguments for Stream.

    """

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(data=data, **kwargs)

if __name__ == "__main__":
    # Placeholder for main execution code
    ...
