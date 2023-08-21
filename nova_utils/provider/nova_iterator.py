import sys
import numpy as np
import os
import copy
import warnings

from typing import Union
from nova_utils.data.data import Data
from nova_utils.data.stream import Stream, StreamMetaData
from nova_utils.data.annotation import Annotation, AnnoMetaData

from pathlib import Path
from nova_utils.data.handler.mongo_handler import AnnotationHandler, StreamHandler
from nova_utils.data.handler.file_handler import FileHandler
from nova_utils.utils import string_utils


class Session:

    def __init__(self, data: dict, dataset: str, name: str, duration: int):
        self.data = data
        self.dataset = dataset
        self.name = name
        self.duration = duration


class NovaIterator:
    def __init__(
            self,
            # Database connection
            ip: str,
            port: int,
            user: str,
            password: str,
            dataset: str,
            data_dir: Path = None,

            # Data
            sessions: list[str] = None,
            data: list[dict] = None,

            # Iterator Window
            frame_size: Union[int , float , str] = None,
            start: Union[int , float , str] = None,
            end: Union[int , float , str] = None,
            left_context: Union[int , float , str] = None,
            right_context: Union[int , float , str] = None,
            stride: Union[int , float , str] = None,

            # Iterator properties
            add_rest_class: bool = True,
            fill_missing_data = True

        ):


        self.data_dir = data_dir
        self.dataset = dataset
        self.sessions = sessions
        self.data = data


        # If stride has not been explicitly set it's the same as the frame size
        if stride is None:
            self.stride = frame_size

        # Parse all times to milliseconds
        self.left_context = string_utils.parse_time_string_to_ms(left_context)
        self.right_context = string_utils.parse_time_string_to_ms(right_context)
        self.frame_size = string_utils.parse_time_string_to_ms(frame_size)
        self.stride = string_utils.parse_time_string_to_ms(stride)
        self.start = string_utils.parse_time_string_to_ms(start)
        self.end = string_utils.parse_time_string_to_ms(end)


        # Frame size 0 or None indicates that the whole session should be returned as one sample
        if self.frame_size == 0:
            warnings.warn("WARNING: Frame size should be bigger than zero. Returning whole session as sample.")

        # If the end time has not been set we initialize it with sys.maxsize
        if self.end is None or self.end == 0:
            self.end = sys.maxsize


        self.add_rest_class = add_rest_class
        self.fill_missing_data = fill_missing_data
        self.current_session = None

        # Data handler
        self._db_anno_handler = AnnotationHandler(ip, port, user, password)
        self._db_stream_handler = StreamHandler(ip, port, user, password, data_dir)
        self._file_handler = FileHandler()

        self._iterable = self._yield_sample()


    def _init_data_from_description(self, data_desc: dict, dataset, session) -> Data:
        src, type_ = data_desc['src'].split(':')
        if src == 'db':
            if type_ == 'anno':
                return self._db_anno_handler.load(dataset=dataset, session=session,scheme=data_desc['scheme'],annotator=data_desc['annotator'],role=data_desc['role'])
            elif type_ == 'stream':
                return self._db_stream_handler.load(dataset=dataset, session=session,name=data_desc['name'], role=data_desc['role'])
            else:
                raise ValueError(f'Unknown data type {type_} for data.')
        elif src == 'file':
            return self._file_handler.load(fp=Path(data_desc['fp']))
        else:
            raise ValueError(f'Unknown source type {src} for data.')

    def _data_description_to_string(self, data_desc:dict) -> str:
        src, type_ = data_desc['src'].split(':')
        delim = '_'
        if src == 'db':
            if type_ == 'anno':
                return delim.join([data_desc['scheme'], data_desc['annotator'],data_desc['role']])
            elif type_ == 'stream':
                return delim.join([data_desc['name'], data_desc['role']])
            else:
                raise ValueError(f'Unknown data type {type_} for data.')
        elif src == 'file':
            return delim.join([data_desc['fp']])
        else:
            raise ValueError(f'Unknown source type {src} for data.')

    def _init_session(self, session):

        # TODO calculate durations properly
        duration = sys.maxsize

        """Opens all annotations and data readers"""
        data = {}

        for data_desc in self.data:
            data_initialized = self._init_data_from_description(data_desc, self.dataset, session)
            data_id = self._data_description_to_string(data_desc)
            data[data_id] = data_initialized

        # for a in self.annos:
        #     annotation = self._db_anno_handler.load(self.dataset, a['scheme'], a['session'], a['annotator'], a['role'])
        #     annotations['_'.join([a['scheme'], a['role'], a['annotator']])] = annotation
        # for s in self.streams:
        #     stream = self._db_stream_handler.load(self.dataset, s['session'], s['role'], s['name'])
        #     streams['_'.join([s['role'], s['annotator']])] = stream

        current_session = Session (
            data = data,
            dataset=self.dataset,
            name = session,
            duration=duration
        )

        return current_session

    def _build_sample_dict(self, labels_for_frame, data_for_frame):
        sample_dict = {}

        garbage_detected = False
        for label_id, label_value in labels_for_frame:
            # if value is not list type check for nan
            if (
                    type(label_value) != list
                    and type(label_value) != str
                    and type(label_value) != np.ndarray
            ):
                if label_value != label_value:
                    garbage_detected = True

            sample_dict.update({label_id: label_value})

        # If at least one label is a garbage label we skip this iteration
        if garbage_detected:
            return None

        for d in data_for_frame:
            sample_dict.update(d)

        # if self.flatten_samples:
        #
        #     # grouping labels and data according to roles
        #     for role in self.roles:
        #         # filter dictionary to contain values for role
        #         sample_dict_for_role = dict(
        #             filter(lambda elem: role in elem[0], sample_dict.items())
        #         )
        #
        #         # remove role from dictionary keys
        #         sample_dict_for_role = dict(
        #             map(
        #                 lambda elem: (elem[0].replace(role + ".", ""), elem[1]),
        #                 sample_dict_for_role.items(),
        #             )
        #         )
        #
        #         sample_dict_for_role["frame"] = (
        #             str(sample_counter) + "_" + key + "_" + role
        #         )
        #         # yield key + '_' + role, sample_dict_for_role
        #         yield sample_dict_for_role
        #         sample_counter += 1
        #     c_pos_ms += _stride_ms
        #
        # else:
        return sample_dict

    def _yield_sample(self):
        """Yields examples."""

        # Needed to sort the samples later and assure that the order is the same as in nova.
        sample_counter = 1

        for session in self.sessions:

            # Init all data objects for the session and get necessary meta information
            self.current_session = self._init_session(session)

            #annotation_dur = [a.meta_data.duration for a in self._annotations]
            #stream_dur = [s.meta_data.duration for s in self._annotations]

            #session_info = self.session_info[session]
            #dur = session_info["duration"]
            #_frame_size_ms = self.frame_size_ms
            #_stride_ms = self.stride_ms

            # If we are loading any datastreams we check if any datastream is shorter than the duration stored in the database suggests
            #if self.data_info:
            #    dur = min(*[v.dur for k, v in self.data_info.items()], dur)

            #if not dur:
            #    raise ValueError("Session {} has no duration.".format(session))

            #dur_ms = int(dur * 1000)

            #If frame size is not specified we return the whole session as one junk
            if self.frame_size <= 0:
               _frame_size = min(self.current_session.duration, self.end - self.start)
               _stride_ms = _frame_size
            else:
                _frame_size = self.frame_size

            # Starting position of the first frame in seconds
            c_pos_ms = max(self.left_context, self.start)

            # TODO account for strid and framesize being None
            # Generate samples for this session
            while c_pos_ms + self.stride + self.right_context <= min(
                    self.end, self.current_session.duration
            ):

                frame_start = c_pos_ms
                frame_end = c_pos_ms + _frame_size

                window_start = frame_start - self.left_context
                window_end = frame_end + self.right_context


                window_info = (
                        session
                        + "_"
                        + str(window_start / 1000)
                        + "_"
                        + str(window_end / 1000)
                )


                # Get data based on frame
                data_for_window = {
                    k : v.sample_from_interval(window_start, window_end) for k, v in self.current_session.data.items()
                }

                # labels_for_window = [
                #     (k, v.get_label_for_frame(frame_start_ms, frame_end_ms))
                #     for k, v in self.current_session.data
                # ]
                #
                # # Get data based on window
                # data_for_window = []
                # for k, v in self.data_info.items():
                #     sample = v.get_sample(window_start, window_end)
                #     # TODO: Special case with empty streams is probably not working correctly. Verify
                #     if sample.shape[0] == 0:
                #         print(f"Sample{window_start}-{window_end} is empty")
                #         #c_pos_ms += _stride_ms
                #         #continue
                #     data_for_window.append({k: sample})

                sample_dict = self._build_sample_dict(data_for_window)
                if not sample_dict:
                    c_pos_ms += _stride_ms
                    sample_counter += 1
                    continue

                yield sample_dict
                c_pos_ms += _stride_ms
                sample_counter += 1

    def __iter__(self):
        return self._iterable

    def __next__(self):
        return self._iterable.__next__()

    # def get_output_info(self):
    #     def map_label_id(lid):
    #         if self.flatten_samples and not lid == "frame":
    #             return split_role_key(lid)[-1]
    #         return lid
    #
    #     return {
    #         # Adding fake framenumber label for sorting
    #         "frame": {"dtype": np.str, "shape": (1,)},
    #         **{map_label_id(k): v.get_info()[1] for k, v in self.annos.items()},
    #         **{map_label_id(k): v.get_info()[1] for k, v in self.data_info.items()},
    # }


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv("../../.env")
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")
    DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", None))

    dataset = 'test'
    sessions = ['04_Oesterreich_test']

    annotation = {
        'src': 'db:anno',
        'scheme': 'transcript',
        'annotator': 'whisperx',
        'role': 'testrole'
    }

    stream = {
        'src': 'db:stream',
        'role': 'testrole',
        'name': 'arousal.synchrony[testrole2]'
    }

    file = {
        'src' : 'file:stream',
        'fp': '/Users/dominikschiller/Work/github/nova-utils/test_files/new_test_video.mp4'
    }

    nova_iterator = NovaIterator(
        IP,
        PORT,
        USER,
        PASSWORD,
        dataset,
        DATA_DIR,
        sessions=sessions,
        data = [annotation, stream, file],
        frame_size='1s',
        end = '5s'
    )

    a = next(nova_iterator)
    breakpoint()