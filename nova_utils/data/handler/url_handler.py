""" Module for handling File data operations related to annotations and streams.

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    24.10.2023

"""
from pathlib import Path
from nova_utils.data.data import Data
from nova_utils.data.handler.ihandler import IHandler
from nova_utils.data.handler.file_handler import FileHandler
from nova_utils.utils.cache_utils import retreive_from_url

class URLHandler(IHandler):
    """Class for handling different types of data files."""

    def __init__(self, download_dir: int = None):
        self.download_dir = download_dir

    def load(self, url: str) -> Data:
        """
        Load data from a url.
        Args:
            fp (Path): The file path.
            header_only (bool): If true only the stream header will be loaded.

        Returns:
            Data: The loaded data.
        """
        fp, _ = retreive_from_url(url)
        fh = FileHandler()
        data = fh.load(Path(fp))
        return data

    def save(self, *args, **kwargs):
        raise NotImplementedError


if __name__ == '__main__':
    uh = URLHandler()
    data = uh.load('https://download.samplelib.com/mp4/sample-5s.mp4')
    breakpoint()