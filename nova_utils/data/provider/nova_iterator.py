import sys
import os
import warnings

from typing import Union
from nova_utils.data.data import Data
from nova_utils.data.stream import Stream, StreamMetaData
from pathlib import Path
from nova_utils.data.handler.mongo_handler import AnnotationHandler, StreamHandler, SessionHandler
from nova_utils.data.handler.file_handler import FileHandler
from nova_utils.utils import string_utils
from nova_utils.utils.anno_utils import data_contains_garbage
from nova_utils.data.session import Session


class NovaIterator:
    """Iterator class for processing data samples from the Nova dataset.

    The NovaIterator takes all information about what data should be loaded and how it should be processed. The class itself then takes care of loading all data and provides an iterator to directly apply a sliding window to the requested data.
    Every time based argument can be passed either as string or a numerical value. If the time is passed as string, the string should end with either 's' to indicate the time is specified in seconds or 'ms' for milliseconds.
    If the time is passed as a numerical value or as a string without indicating a specific unit it is assumed that an integer value represents milliseconds while a float represents seconds. All numbers will be represented as integer milliseconds internally.
    The highest time resolution for processing is therefore 1ms.

    Args:
        db_host (str): Database IP address.
        db_port (int): Database port.
        db_user (str): Database username.
        db_password (str): Database password.
        dataset (str): Name of the dataset.
        data_dir (Path, optional): Path to the data directory. Defaults to None.
        sessions (list[str], optional): List of session names to process. Defaults to None.
        data (list[dict], optional): List of data descriptions. Defaults to None.
        frame_size (Union[int, float, str], optional): Size of the data frame measured in time. Defaults to None.
        start (Union[int, float, str], optional): Start time for processing measured in time. Defaults to None.
        end (Union[int, float, str], optional): End time for processing measured in time. Defaults to None.
        left_context (Union[int, float, str], optional): Left context duration measured in time. Defaults to None.
        right_context (Union[int, float, str], optional): Right context duration measured in time. Defaults to None.
        stride (Union[int, float, str], optional): Stride for iterating over data measured in time. If stride is not set explicitly it will be set to frame_size. Defaults to None.
        add_rest_class (bool, optional): Whether to add a rest class for discrete annotations. Defaults to True.
        fill_missing_data (bool, optional): Whether to fill missing data. Defaults to True. THIS OPTION IS CURRENTLY NOT DOING ANYTHING

    Attributes:
        data_dir (Path): Path to the data directory.
        dataset (str): Name of the dataset.
        sessions (list[str]): List of session names to process.
        data (list[dict]): List of data descriptions.
        ...
    """
    def __init__(
            self,
            # Database connection
            db_host: str,
            db_port: int,
            db_user: str,
            db_password: str,
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
            warnings.warn("Frame size should be bigger than zero. Returning whole session as sample.")

        # If the end time has not been set we initialize it with sys.maxsize
        if self.end is None or self.end == 0:
            self.end = sys.maxsize


        self.add_rest_class = add_rest_class
        self.fill_missing_data = fill_missing_data
        self.current_session = None

        # Data handler
        self._db_session_handler = SessionHandler(db_host, db_port, db_user, db_password)
        self._db_anno_handler = AnnotationHandler(db_host, db_port, db_user, db_password)
        self._db_stream_handler = StreamHandler(db_host, db_port, db_user, db_password, data_dir=data_dir)
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

    def _init_session(self, session_name: str) -> Session:

        session = self._db_session_handler.load(self.dataset, session_name)

        """Opens all annotations and data readers"""
        data = {}

        # setting session data
        for data_desc in self.data:
            data_initialized = self._init_data_from_description(data_desc, self.dataset, session_name)
            data_id = self._data_description_to_string(data_desc)
            data[data_id] = data_initialized
        session.data = data

        # update session duration
        min_dur = session.duration if session.duration is not None else sys.maxsize
        for data_initialized in data.values():
            if isinstance(data_initialized, Stream):
                meta_data : StreamMetaData = data_initialized.meta_data
                if meta_data.duration is not None:
                    dur = meta_data.duration
                else:
                    dur = len(data_initialized.data) / meta_data.sample_rate * 1000
                if dur < min_dur:
                    min_dur = dur
        session.duration = min_dur

        if session.duration == sys.maxsize:
            raise ValueError(f'Unable to determine duration for session {session.name}')

        return session


    def _yield_sample(self):
        """Yields examples."""

        # Needed to sort the samples later and assure that the order is the same as in nova.
        sample_counter = 1

        for session in self.sessions:

            # Init all data objects for the session and get necessary meta information
            self.current_session = self._init_session(session)

            #If frame size is zero or less we return the whole data from the whole session in one sample
            if self.frame_size <= 0:
               _frame_size = min(self.current_session.duration, self.end - self.start)
               _stride = _frame_size
            else:
                _frame_size = self.frame_size
                _stride = self.stride

            # Starting position of the first frame in seconds
            cpos = max(self.left_context, self.start)

            # TODO account for stride and framesize being None
            # Generate samples for this session
            while cpos + self.stride + self.right_context <= min(
                    self.end, self.current_session.duration
            ):

                frame_start = cpos
                frame_end = cpos + _frame_size
                window_start = frame_start - self.left_context
                window_end = frame_end + self.right_context


                window_info = (
                        session
                        + "_"
                        + str(window_start / 1000)
                        + "_"
                        + str(window_end / 1000)
                )

                # Load data for frame
                data_for_window = {
                    k : v.sample_from_interval(window_start, window_end) for k, v in self.current_session.data.items()
                }

                # Performing sanity checks
                garbage_detected = any([ data_contains_garbage(d) for k, d in data_for_window.items()  ])

                # Incrementing counter
                cpos += _stride
                sample_counter += 1

                if garbage_detected:
                    continue

                yield data_for_window


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

    load_dotenv("../../../.env")
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
        data = [annotation, file],
        frame_size='5s',
        end = '20s'
    )

    a = next(nova_iterator)
    breakpoint()