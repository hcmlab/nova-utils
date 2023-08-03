from idata import IDynamicData
class AudioData(IDynamicData):
    ...
class VideoData(IDynamicData):
    def __init__(self, *args, **kwargs):
        pass
class StreamData(IDynamicData):
    ...


# from decord import VideoReader, AudioReader, cpu
# import numpy as np
# import os
# import sys
# import errno
# import math
#
# from pathlib import Path
# from nova_utils.deprecated_db_utils import nova_types as nt
# from nova_utils.deprecated_ssi_utils.ssi_stream_utils import Stream
# from hcai_datasets.hcai_nova_dynamic.utils.nova_string_utils import (
#     merge_role_key,
# )
# from typing import Union
# from abc import ABC, abstractmethod
# from enum import Enum
#
#
# class StrideUnit(Enum):
#     MILLISECONDS = (0,)
#     SECONDS = (1,)
#     FRAMES = 2
#
#
# class Data(ABC):
#     def __init__(
#             self,
#             role: str = "",
#             name: str = "",
#             file_ext: str = "stream",
#             sr: int = 0,
#             data_type: nt.DataTypes = None,
#             np_data_type: np.dtype = np.float32,
#             is_valid: bool = True,
#             sample_data_path: str = "",
#             sample_data_shape: tuple = None,
#             lazy_loading: bool = False,
#     ):
#         """
#         Abstract base class for all data types
#         Args:
#             role (str): Role to which the datastream belongs
#             name (str): Filename of the datastream without the role
#             file_ext (str): File extension of the datastream
#             sr (int): Samplerate of the datastream
#             data_type (DataTypes): Type of the datastream as specified in the nova_utils package under db_utils.nova_types.novDataTypes
#             is_valid (bool): Whether the datastream is valid or not
#             sample_data_path (str): A path to an example data file from which all meta information that is not saved in the Nova database can be read
#             sample_data_shape (tuple): The default shape of one data sample. Will be overwritten by populate_meta_info if a valid datastream can be found.
#             lazy_loading (bool): If set to true only the respective timestamps and filenames will be returned for each window. Not the actual data.
#         """
#         self.role = role
#         self.name = name
#         self.is_valid = is_valid
#         self.sr = sr
#         self.file_ext = file_ext
#         self.lazy_loading = lazy_loading
#         self.data_type = data_type
#         self.np_data_type = np_data_type
#
#         # Set when populate_meta_info is called
#         self.sample_data_shape = sample_data_shape
#         self.meta_loaded = False
#         # self.n_frames_per_window = n_samples_per_window
#
#         # Set when open_file_reader is called
#         self.file_path = None
#         self.file_reader = None
#         self.dur = sys.maxsize
#
#         if sample_data_path:
#             if not os.path.isfile(sample_data_path):
#                 self.meta_loaded = False
#                 print(
#                     f"WARNING: Sample file {sample_data_path} not found. Could not initialize meta data."
#                 )
#                 # raise FileNotFoundError( errno.ENOENT, os.strerror(errno.ENOENT), sample_data_path)
#             else:
#                 self.populate_meta_info(sample_data_path)
#
#     def data_stream_opend(self):
#         if not self.file_reader:
#             print(
#                 "No datastream opened for {}".format(
#                     merge_role_key(self.role, self.name)
#                 )
#             )
#             raise RuntimeError("Datastream not loaded")
#
#     def get_info(self):
#         if self.meta_loaded:
#             if self.lazy_loading:
#                 return {
#                     "frame_start": {"dtype": np.float32, "shape": (1)},
#                     "frame_end": {"dtype": np.float32, "shape": (1)},
#                     "file_path": {"dtype": np.str, "shape": (None,)},
#                 }
#             return self._get_info_hook()
#         else:
#             print(
#                 "Meta data has not been loaded for file {}. Call get_meta_info() first.".format(
#                     merge_role_key(self.role, self.name)
#                 )
#             )
#
#     def get_sample(
#             self,
#             window_start: int,
#             window_end: int,
#             stride_unit: StrideUnit = StrideUnit.MILLISECONDS,
#     ):
#         """
#         Returns the sample for the respective frames. If lazy loading is true, only the filepath and frame_start, frame_end will be returned.
#         """
#
#         if not self.file_reader:
#             start_frame, end_frame = sample_window_from_time_interval(
#                 window_start, window_end, self.sr
#             )
#             return np.zeros((end_frame - start_frame,) + self.sample_data_shape).astype(self.np_data_type)
#         elif self.lazy_loading:
#             return {
#                 "frame_start": window_start,
#                 "frame_end": window_end,
#                 "file_path": self.file_path,
#             }
#         else:
#             return self._get_sample_hook(window_start, window_end, stride_unit)
#
#     def open_file_reader(self, path: str):
#         """
#         Args:
#             path ():
#         Raises:
#             FileNotFoundError: If path is not a file
#         """
#         self.file_path = path
#         if not os.path.isfile(path):
#             raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
#         if not self.meta_loaded:
#             self.populate_meta_info(path)
#         self._open_file_reader_hook(path)
#
#     @abstractmethod
#     def _get_info_hook(self):
#         """
#         Returns the features for this datastream to create the DatasetInfo for tensorflow
#         """
#         ...
#
#     @abstractmethod
#     def _get_sample_hook(
#             self,
#             window_start: int,
#             window_end: int,
#             stride_unit: StrideUnit = StrideUnit.MILLISECONDS,
#     ):
#         """
#         Returns a data chunk from window start to window end. The window is automatically adjusted to always provide the same number of data samples.
#
#         Args:
#             stride_unit ():
#         """
#         ...
#
#     @abstractmethod
#     def _open_file_reader_hook(self, path: str):
#         """
#         Opens a filereader for the respective datastream. Sets attributes self.file_reader and self.dur
#         """
#         ...
#
#     @abstractmethod
#     def populate_meta_info(self, path: str):
#         """
#         Opens a data sample from the provided path to extract additional data that is not in the database
#         """
#         ...
#
#     @abstractmethod
#     def close_file_reader(self):
#         """
#         Closes a filereader for the respective datastream
#         """
#         ...
#
#
# class AudioData(Data):
#     def __init__(self, **kwargs):
#         """
#
#         Args:
#             **kwargs ():
#         """
#         # Overwrite default
#         if kwargs.get("sample_data_shape") is None:
#             sample_data_shape = (1,)
#             kwargs.update({"sample_data_shape": sample_data_shape})
#         if kwargs.get("np_data_type") is None:
#             kwargs.update({"np_data_type": np.float32})
#             kwargs.update({"data_type": nt.DataTypes.AUDIO})
#
#         super().__init__(**kwargs)
#
#     def _get_info_hook(self):
#         return merge_role_key(self.role, self.name), {
#             "shape": self.sample_data_shape,
#             "dtype": np.float32,
#         }
#
#     def _get_sample_hook(
#             self,
#             window_start: int,
#             window_end: int,
#             stride_unit: StrideUnit = StrideUnit.FRAMES,
#     ):
#
#         if stride_unit == StrideUnit.FRAMES:
#             window_start_sample = window_start
#             window_end_sample = window_end
#         else:
#             window_start_sample, window_end_sample = sample_window_from_time_interval(
#                 window_start, window_end, self.sr
#             )
#         chunk = self.file_reader.get_batch(
#             list(range(window_start_sample, window_end_sample))
#         ).asnumpy()
#         return chunk
#
#     def _open_file_reader_hook(self, path: str):
#         self.file_reader = AudioReader(path, ctx=cpu(0), mono=False)
#         self.dur = self.file_reader.duration()
#         self.n_frames = len(self.file_reader)
#
#     def populate_meta_info(self, path: str):
#         """
#
#         Args:
#             path ():
#         """
#
#         file_reader = AudioReader(path, ctx=cpu(0), mono=False)
#         n_channels = file_reader.shape[0]
#         self.sample_data_shape = (n_channels, None)
#         self.meta_loaded = True
#
#     def close_file_reader(self):
#         return True
#
#
# class VideoData(Data):
#     def __init__(self, **kwargs: object) -> object:
#         """
#
#         Args:
#             **kwargs ():
#         """
#         # Overwrite default
#         if kwargs.get("sample_data_shape") is None:
#             sample_data_shape = (480, 640, 3)
#             kwargs.update({"sample_data_shape": sample_data_shape})
#         if kwargs.get("np_data_type") is None:
#             kwargs.update({"np_data_type": np.uint8})
#         kwargs.update({"data_type": nt.DataTypes.VIDEO})
#
#         super().__init__(**kwargs)
#
#     def _get_info_hook(self):
#         return merge_role_key(self.role, self.name), {
#             "shape": self.sample_data_shape,
#             "dtype": np.float32,
#         }
#
#     def _get_sample_hook(
#             self,
#             window_start: int,
#             window_end: int,
#             stride_unit: StrideUnit = StrideUnit.FRAMES,
#     ):
#         if stride_unit == StrideUnit.FRAMES:
#             window_start_sample = window_start
#             window_end_sample = window_end
#         else:
#             window_start_sample, window_end_sample = sample_window_from_time_interval(
#                 window_start, window_end, self.sr
#             )
#
#         chunk = self.file_reader.get_batch(
#             list(range(window_start_sample, window_end_sample))
#         ).asnumpy()
#         return chunk
#
#     def _open_file_reader_hook(self, path: str):
#         self.file_reader = VideoReader(path, ctx=cpu(0))
#         fps = self.file_reader.get_avg_fps()
#         frame_count = len(self.file_reader)
#         self.dur = frame_count / fps
#         self.n_frames = len(self.file_reader)
#
#     def populate_meta_info(self, path: str):
#         file_reader = VideoReader(path)
#         self.sample_data_shape = file_reader[0].shape
#         self.meta_loaded = True
#
#     def close_file_reader(self):
#         return True
#
#
# class StreamData(Data):
#     def __init__(self, **kwargs: object) -> object:
#         """
#
#         Args:
#             **kwargs ():
#         """
#         # Overwrite default
#         if kwargs.get("sample_data_shape") is None:
#             sample_data_shape = (1,)
#             kwargs.update({"sample_data_shape": sample_data_shape})
#         if kwargs.get("np_data_type") is None:
#             kwargs.update({"np_data_type": np.float32})
#         kwargs.update({"data_type": nt.DataTypes.FEATURE})
#
#         super().__init__(**kwargs)
#
#     def _get_info_hook(self):
#         return merge_role_key(self.role, self.name), {
#             "shape": self.sample_data_shape,
#             "dtype": self.np_data_type,
#         }
#
#     def _get_sample_hook(
#             self,
#             window_start: int,
#             window_end: int,
#             stride_unit: StrideUnit = StrideUnit.FRAMES,
#     ):
#         try:
#             self.data_stream_opend()
#             if stride_unit == StrideUnit.FRAMES:
#                 window_start_sample = window_start
#                 window_end_sample = window_end
#             else:
#                 (
#                     window_start_sample,
#                     window_end_sample,
#                 ) = sample_window_from_time_interval(window_start, window_end, self.sr)
#
#             return self.file_reader.data[window_start_sample:window_end_sample]
#         except RuntimeError:
#             print(
#                 "Could not get chunk {}-{} from data stream {}".format(
#                     window_start, window_end, merge_role_key(self.role, self.name)
#                 )
#             )
#
#     def _open_file_reader_hook(self, path: Path) -> bool:
#         stream = Stream(path=path)
#         if stream:
#             self.file_reader = stream
#             self.n_frames = stream.data.shape[0]
#             self.dur = self.n_frames / stream.sr
#             return True
#         else:
#             print("Could not open Stream {}".format(str))
#             return False
#
#     def close_file_reader(self):
#         return True
#
#     def populate_meta_info(self, path: str):
#         stream = Stream()
#         stream.load_header(path)
#         self.sample_data_shape = (stream.dim,)
#         self.np_data_type = stream.dtype
#         self.meta_loaded = True
#
#
# ##########################
# # General helper functions
# ##########################
#
#
# # def frame_to_seconds(sr: int, frame: int) -> float:
# #    return frame / sr
#
#
# def seconds_to_sample_nr(time_s: float, sr: float) -> float:
#     """
#     Calculates the specific sample number in a data stream that corresponds to a given time
#     Args:
#         sr (float): The sample rate of the data stream
#         time_s (float): The timestamp of the sample in seconds
#
#     Returns:
#         float: Unrounded index of the sample in the stream for a given time
#
#     """
#     return time_s * sr
#
#
# def milli_seconds_to_sample_nr(time_ms: int, sr: float) -> float:
#     """
#     Calculates the specific sample number in a data stream that corresponds to a given time
#     Args:
#         sr (float): The sample rate of the data stream
#         time_ms (int): The timestamp of the sample in milliseconds
#
#     Returns:
#         float: Unrounded index of the sample in the stream for a given time
#
#     """
#     return seconds_to_sample_nr(sr=sr, time_s=time_ms / 1000.0)
#
#
# def sample_nr_to_seconds(sample_nr: float, sr: float) -> float:
#     """
#     Calculates the specific time in seconds in a data stream that corresponds to given sample number
#     Args:
#         sample_nr (float): The unrounded sample number. Might represent a value between to actual samples of the stream.
#         sr (float): The average numbers of samples per second.
#
#     Returns:
#         float: Unrounded time representation of sample_nr specified in seconds
#     """
#     return sample_nr * sr
#
#
# def sample_nr_to_milli_seconds(sample_nr: float, sr: float) -> float:
#     """
#     Calculates the specific time in milli in a data stream that corresponds to given sample number
#     Args:
#         sample_nr (float): The unrounded sample number. Might represent a value between to actual samples of the stream.
#         sr (float): The average numbers of samples per second.
#
#      Returns:
#          float: Unrounded time representation of sample_nr specified in milliseconds
#     """
#     return sample_nr_to_seconds(sample_nr, sr) * 1000.0
#
#
# def sample_window_from_time_interval(
#         start_time_ms: int, end_time_ms: int, sr: float
# ) -> tuple[int, int]:
#     """
#     Calculates the start and end sample number of a sliding window.
#     Args:
#         start_time_ms (int): Start time of the window in milliseconds
#         end_time_ms (int): End time of the window in milliseconds
#         sr (float): Sample rate of the data stream
#
#     Returns:
#         tuple[int, int]: First and last sample number of the window
#
#     """
#
#     # Get unrounded sample numbers
#     start_sample_nr = milli_seconds_to_sample_nr(start_time_ms, sr)
#     end_sample_nr = milli_seconds_to_sample_nr(end_time_ms, sr)
#
#     # Rounding start and end to include maximum information
#     start_sample_nr = math.floor(start_sample_nr)
#     end_sample_nr = math.ceil(end_sample_nr)
#
#     # Assert number of samples in window
#     expected_number_of_samples = math.ceil(((end_time_ms - start_time_ms) / 1000) * sr)
#     number_of_samples = end_sample_nr - start_sample_nr
#     sample_diff = number_of_samples - expected_number_of_samples
#
#     # Fewer then expected samples: We move start_sample number to the left
#     # More samples than expected: We move start_sample number to the right
#     start_sample_nr += sample_diff
#
#     assert expected_number_of_samples == end_sample_nr - start_sample_nr
#
#     return start_sample_nr, end_sample_nr
#
#
# def parse_time_string_to_ms(frame: Union[str, int, float]) -> int:
#     # if frame is specified milliseconds as string
#     if str(frame).endswith("ms"):
#         try:
#             return int(frame[:-2])
#         except ValueError:
#             raise ValueError(
#                 "Invalid input format for frame in milliseconds: {}".format(frame)
#             )
#     # if frame is specified in seconds as string
#     elif str(frame).endswith("s"):
#         try:
#             frame_s = float(frame[:-1])
#             return int(frame_s * 1000)
#         except ValueError:
#             raise ValueError(
#                 "Invalid input format for frame in seconds: {}".format(frame)
#             )
#     # if type is float we assume the input will be seconds
#     elif isinstance(frame, float) or "." in str(frame):
#         try:
#             print(
#                 "WARNING: Automatically inferred type for frame {} is float.".format(
#                     frame
#                 )
#             )
#             return int(1000 * float(frame))
#         except ValueError:
#             raise ValueError("Invalid input format for frame: {}".format(frame))
#     # if type is int we assume the input will be milliseconds
#     elif isinstance(frame, int) or (isinstance(frame, str) and frame.isdigit()):
#         try:
#             print(
#                 "WARNING: Automatically inferred type for frame {} is int.".format(
#                     frame
#                 )
#             )
#             return int(frame)
#         except ValueError:
#             raise ValueError("Invalid input format for frame: {}".format(frame))
#
#     print(
#         f'WARNING: Could  not automatically parse time "{frame}" to seconds. Returning None '
#     )
#
#
# def parse_time_str(frame: Union[str, int, float]) -> (Union[int, float], StrideUnit):
#
#     if frame is None:
#         print(f"WARNING: No timestring to parse. Returning None ")
#         return None, StrideUnit.MILLISECONDS
#
#     # if frame is specified milliseconds
#     if str(frame).endswith("ms"):
#         try:
#             return int(frame[:-2]), StrideUnit.MILLISECONDS
#         except ValueError:
#             raise ValueError(
#                 "Invalid input format for length specified in milliseconds: {}".format(
#                     frame
#                 )
#             )
#
#     # if frame is specified in seconds
#     elif str(frame).endswith("s"):
#         try:
#             return float(frame[:-1]), StrideUnit.SECONDS
#         except ValueError:
#             raise ValueError(
#                 "Invalid input format for length specified in seconds: {}".format(frame)
#             )
#
#     # if frame is specified in frames
#     elif str(frame).endswith("f"):
#         try:
#             return int(frame[:-1]), StrideUnit.FRAMES
#         except ValueError:
#             raise ValueError(
#                 "Invalid input format for length specified in number of frames: {}".format(
#                     frame
#                 )
#             )
#     # if type is float we assume the input will be seconds
#     elif isinstance(frame, float) or "." in str(frame):
#         try:
#             print(
#                 "WARNING: Automatically inferred type for frame {} is float. Interpreting as seconds".format(
#                     frame
#                 )
#             )
#             return float(frame), StrideUnit.SECONDS
#         except ValueError:
#             raise ValueError("Invalid input format for frame: {}".format(frame))
#     # if type is int we assume the input will be milliseconds
#     elif isinstance(frame, int) or (isinstance(frame, str) and frame.isdigit()):
#         try:
#             print(
#                 "WARNING: Automatically inferred type for frame {} is int. Interpreting as milliseconds".format(
#                     frame
#                 )
#             )
#             return int(frame), StrideUnit.MILLISECONDS
#         except ValueError:
#             raise ValueError("Invalid input format for frame: {}".format(frame))
#
#     print(f'WARNING: Could  not automatically parse time "{frame}". Returning None ')
#     return None, StrideUnit.MILLISECONDS
