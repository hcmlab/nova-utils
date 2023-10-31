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
        output_name = ''
        retreive_from_url(url, )
        return data

    def save(self, *args, **kwargs):
        raise NotImplementedError


if __name__ == "__main__":
    # Test cases...
    test_annotations = True
    test_streams = True
    base_dir = Path("../../../test_files/")
    fh = FileHandler()

    """TESTCASE FOR ANNOTATIONS"""
    if test_annotations:

        # ascii read
        discrete_anno_ascii = fh.load(base_dir / "discrete_ascii.annotation")
        continuous_anno_ascii = fh.load(base_dir / "continuous_ascii.annotation")
        free_anno_ascii = fh.load(base_dir / "free_ascii.annotation")

        # binary read
        discrete_anno_binary = fh.load(base_dir / "discrete_binary.annotation")
        continuous_anno_binary = fh.load(base_dir / "continuous_binary.annotation")

        # ascii write
        fh.save(discrete_anno_ascii, base_dir / "new_discrete_ascii.annotation")
        fh.save(continuous_anno_ascii, base_dir / "new_continuous_ascii.annotation")
        fh.save(free_anno_ascii, base_dir / "new_free_ascii.annotation")

        # binary write
        fh.save(
            discrete_anno_binary,
            base_dir / "new_discrete_binary.annotation",
            ftype=SSIFileType.BINARY,
            )
        fh.save(
            continuous_anno_binary,
            base_dir / "new_continuous_binary.annotation",
            ftype=SSIFileType.BINARY,
            )

        # verify
        discrete_anno_ascii_new = fh.load(base_dir / "new_discrete_ascii.annotation")
        continuous_anno_ascii_new = fh.load(
            base_dir / "new_continuous_ascii.annotation"
        )
        free_anno_ascii_new = fh.load(base_dir / "new_free_ascii.annotation")

        # binary read
        discrete_anno_binary_new = fh.load(base_dir / "new_discrete_binary.annotation")
        continuous_anno_binary_new = fh.load(
            base_dir / "new_continuous_binary.annotation"
        )

    """TESTCASE FOR STREAMS"""
    if test_streams:

        # ssistream read
        ssistream_ascii = fh.load(base_dir / "ascii.stream")
        ssistream_binary = fh.load(base_dir / "binary.stream")

        # Replace one dimension with random data
        new_data = ssistream_binary.data.copy()
        replacement_dimension = 0
        random_data = np.random.rand(new_data.shape[replacement_dimension])

        # Generate random data
        new_data[:, replacement_dimension] = random_data
        ssistream_binary.data = new_data
        ssistream_ascii.data = new_data

        # ssistream write
        fh.save(ssistream_ascii, base_dir / "new_ascii.stream", SSIFileType.ASCII)
        fh.save(ssistream_binary, base_dir / "new_binary.stream", SSIFileType.BINARY)

        # audio
        audio = fh.load(base_dir / "test_audio.wav")

        fh.save(audio, base_dir / "new_test_audio.wav")

        new_audio = fh.load(base_dir / "new_test_audio.wav")

        np.allclose(audio.data[0:10000], new_audio.data[0:10000])

        # video
        video = fh.load(base_dir / "test_video.mp4")

        fh.save(video, base_dir / "new_test_video.mp4")

        new_video = fh.load(base_dir / "new_test_video.mp4")

        assert new_video.data[0:30].all() == video.data[0:30].all()
        breakpoint()
