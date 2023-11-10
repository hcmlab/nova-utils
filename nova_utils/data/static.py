"""Definition of all Stream classes and Metadata
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 18.8.2023
"""

import numpy as np
from typing import Union
from nova_utils.data.data import Data


# TODO Refactoring: StaticMetaData is currently a subset of StreamMetaData. This common properties should be moved to a parent class
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
            sample_shape: tuple = None,
            dtype: np.dtype = None,
            custom_meta: dict = None
    ):
        """
        Initialize a StreamMetaData instance with stream properties.
        """
        self.name = name
        self.ext = ext
        self.sample_shape = sample_shape
        self.dtype = dtype
        self.custom_meta = custom_meta if custom_meta is not None else {}

class StaticData(Data):
    """
    A subclass of Data representing static data.
    (No additional methods or attributes specified in the provided code. At the moment this is just a placeholder class.)
    """
    def __init__(
            self,
            data: Union[np.ndarray, None],
            name: str = None,
            ext: str = None,
            sample_shape: tuple = None,
            dtype: np.dtype = None,
            custom_meta: dict = None,
            **kwargs
    ):
        super().__init__(data=data, **kwargs)
        # Add Metadata
        if ext is None:
            if isinstance(self, Image):
                ext = '.jpg'
            if isinstance(self, Text):
                ext = '.text'

        static_meta_data = StaticMetaData(name, ext, sample_shape, dtype, custom_meta)
        self.meta_data.expand(static_meta_data)


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
