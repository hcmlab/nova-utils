
video_path = 'test_files/test_video.mp4'

import mmap
import numpy as np
import decord

class LazyVideoArray(np.ndarray):
    def __new__(cls, video_reader, dtype=None, shape=None, buffer=None, offset=0, strides=None):
        if dtype is None:
            dtype = np.uint8
        if shape is None:
            shape = (len(video_reader), 1280, 720, 3)

        if buffer is None:
            buffer = mmap.mmap(-1, 8 * np.prod(shape), access=mmap.ACCESS_WRITE)

        obj = super().__new__(cls, shape, dtype=dtype, buffer=buffer, offset=offset, strides=strides)
        obj.video_reader = video_reader
        return obj

    def __getitem__(self, index):
        if isinstance(index, int):
            frame = self.video_reader[index].asnumpy()
            self[index] = frame  # Cache the loaded frame
            return frame
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.shape[0])
            frames = [self[i] for i in range(start, stop, step)]
            return np.stack(frames)
        else:
            raise ValueError("Unsupported index type")

# Path to your video file

# Open the video file using decord
video_reader = decord.VideoReader(video_path)

# Create a LazyVideoArray instance
lazy_video_array = LazyVideoArray(video_reader, dtype=np.uint8)

# Access frames lazily through the LazyVideoArray
for idx in range(len(lazy_video_array)):
    frame = lazy_video_array[idx]
    # You can process the frame here as needed

# Print a portion of the LazyVideoArray
print("LazyVideoArray Data:")
print(lazy_video_array[0])

# At this point, the LazyVideoArray has the video frames loaded lazily.
