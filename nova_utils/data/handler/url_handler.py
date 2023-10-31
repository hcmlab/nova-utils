""" Module for handling File data operations related to annotations and streams.

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    24.10.2023

"""
from pathlib import Path

import numpy as np

from nova_utils.data.data import Data
from nova_utils.data.handler.ihandler import IHandler
from nova_utils.utils.cache_utils import retreive_from_url
from nova_utils.utils.type_definitions import (
    SSIFileType,
)


class URLHandler(IHandler):
    """Class for handling different types of data files."""

    def __init__(self, download_dir: int = None):
        self.download_dir = download_dir

    def load(self, url: str) -> Data:
        """
        Load data from a file.

        Args:
            fp (Path): The file path.
            header_only (bool): If true only the stream header will be loaded.

        Returns:
            Data: The loaded data.
        """
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

