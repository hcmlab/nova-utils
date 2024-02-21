import numpy as np
import decord
from decord import cpu
class DecordBatchReader(np.ndarray):
    """LazyArray class extending numpy.ndarray for video and audio loading."""
    import decord

    @property
    def shape(self):
        return self._shape

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __new__(cls, filename):
        dr = decord.VideoReader(filename, ctx=cpu(0))
        _f = dr[0]
        dr.seek(0)

        shape = _f.shape[:2]
        dtype = np.dtype(_f.dtype)
        buffer = None

        obj = super().__new__(cls, shape[1:], dtype=dtype, buffer=buffer)
        obj.decord_reader = dr
        obj.start_idx = 0
        obj._num_samples = (shape[0])
        obj.len = dr._num_frame
        obj._shape = shape
        return obj

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = list(range(index.start, index.stop))
            ret = self.decord_reader.get_batch(indices).asnumpy()
            # if type(self.decord_reader) == decord.video_reader.VideoReader:
            #     self.decord_reader.seek_accurate(index.start)
        elif isinstance(index, list):
            ret = np.squeeze(self.decord_reader.get_batch([index]).asnumpy())
            # if type(self.decord_reader) == decord.video_reader.VideoReader:
            #     self.decord_reader.seek(index[0])
        else:
            ret = self.decord_reader[index].asnumpy()
            # if type(self.decord_reader) == decord.video_reader.VideoReader:
            #     self.decord_reader.seek(index)
        return ret