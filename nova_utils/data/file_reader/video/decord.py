import decord
import numpy as np
from decord import cpu
from pims import FramesSequence, Frame

class DecordReader(FramesSequence):

    def __init__(self, filename):
        self.filename = filename
        self.reader = decord.VideoReader(filename, ctx=cpu(0))
        self._len = self.reader._num_frame
        _f = self.reader[0]
        self.reader.seek(0)
        self._dtype = np.dtype(_f.dtype)
        self._frame_shape = _f.shape[:2]

    def get_frame(self, i):
        # Access the data you need and get it into a numpy array.
        # Then return a Frame like so:
        return Frame(self.reader[i].asnumpy(), frame_no=i)

    def __len__(self):
        return self._len

    @property
    def frame_shape(self):
        return self._frame_shape

    @property
    def pixel_type(self):
        return self._dtype