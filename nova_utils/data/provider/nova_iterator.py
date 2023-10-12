""" Module to create a data iterator for streams and annotations

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    18.8.2023

"""
import numpy as np
import sys
import os
import warnings

from typing import Union
from nova_utils.data.data import Data
from nova_utils.data.stream import Stream, StreamMetaData, Audio, Video, SSIStream, DynamicData
from pathlib import Path
from nova_utils.data.handler.mongo_handler import (
    AnnotationHandler,
    StreamHandler,
    SessionHandler,
)
from nova_utils.data.handler.file_handler import FileHandler
from nova_utils.utils import string_utils
from nova_utils.utils.anno_utils import data_contains_garbage
from nova_utils.data.session import Session


class NovaIterator:
    """Iterator class for processing data samples from the Nova dataset.

    The NovaIterator takes all information about what data should be loaded and how it should be processed. The class itself then takes care of loading all data and provides an iterator to directly apply a sliding window to the requested data.
    Every time based argument can be passed either as string or a numerical value. If the time is passed as string, the string should end with either 's' to indicate the time is specified in seconds or 'ms' for milliseconds.
    If the time is passed as a numerical value or as a string without indicating a specific unit it is assumed texthat an integer value represents milliseconds while a float represents seconds. All numbers will be represented as integer milliseconds internally.
    The highest time resolution for processing is therefore 1ms.

    Args:
        db_host (str): Database IP address.
        db_port (int): Database port.
        db_user (str): Database username.
        db_password (str): Database password.
        dataset (str): Name of the dataset.
        data_dir (Path, optional): Path to the data directory. Defaults to None.
        sessions (list[str], optional): List of session names to process. Defaults to None.
        data (list[dict[str, str]], optional): List of data descriptions. Defaults to None. The dictionary should have the following fields:

            ``"id"``:
                Unique id to map the data to a given input / output.
            ``"name"``:
                Output name for streams
            ``"type"``:
                IO type of the data. Either "input" or "output"
            ``"src"``
                The source and datatype to load the data from separated by ':' . Source can be either 'db' for database or 'file' to load from disc.
                The dataytpye is eiter 'stream' or 'anno'. E.g. 'db:anno' .
            ``"scheme"``
                The scheme name of the annotations to load. Only necessary when loading annotations from the database.
            ``"annotator"``
                The annotator of the annotations to load. Only necessary when loading annotations from the database.
            ``"role"``
                The role to which the data belongs. Only necessary when accessing data from the database.
            ``"fp"``
                The filepath from which to load the data from. Only necessary when loading files from disk.

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
        frame_size (int, float, str): Size of the data frame measured in time.
        start (int, float, str): Start time for processing measured in time.
        end (int, float, str): End time for processing measured in time.
        left_context (int, float, str): Left context duration measured in time.
        right_context (int, float, str): Right context duration measured in time.
        stride (int, float, str): Stride for iterating  data measured in time.
        add_rest_class (bool): Whether to add a rest class for discrete annotations.
        fill_missing_data (bool): Whether to fill missing data.

    Example:
        .. code-block:: python

            from dotenv import load_dotenv

            # Load environment variables
            IP = 127.0.0.1
            PORT = 1337
            USER = my_user
            PASSWORD = my_password
            DATA_DIR = /data

            # Define dataset and sessions
            dataset = "test_dataset"
            sessions = ["test_session"]

            # Define data descriptions
            annotation = {
                "id" : "my_transcript_1",
                "type": "input",
                "src": "db:anno",
                "scheme": "transcript",
                "annotator": "test_annotator",
                "role": "test_role",
            }

            stream = {
                "id" : "model_output",
                "type" : "input",
                "src": "db:stream",
                "role": "test_role",
                "name": "extracted_features",
            }

            file = {
                "id": "just_a_file",
                "type": "output",
                "src": "file:stream",
                "fp": "/path/to/my/video/test_video.mp4",
            }

            # Create a NovaIterator instance
            nova_iterator = NovaIterator(
                IP,
                PORT,
                USER,
                PASSWORD,
                dataset,
                DATA_DIR,
                sessions=sessions,
                data=[annotation, file],
                frame_size="5s",
                end="20s",
            )

            # Example: Get the next data sample and set a breakpoint
            a = next(nova_iterator)
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
        frame_size: Union[int, float, str] = None,
        start: Union[int, float, str] = None,
        end: Union[int, float, str] = None,
        left_context: Union[int, float, str] = None,
        right_context: Union[int, float, str] = None,
        stride: Union[int, float, str] = None,
        # Iterator properties
        add_rest_class: bool = True,
        fill_missing_data=True,
    ):
        self.data_dir = data_dir
        self.dataset = dataset
        self.sessions = sessions
        self.data = data

        # If stride has not been explicitly set it's the same as the frame size
        if stride is None:
            stride = frame_size

        # Parse all times to milliseconds
        self.left_context = string_utils.parse_time_string_to_ms(left_context)
        self.right_context = string_utils.parse_time_string_to_ms(right_context)
        self.frame_size = string_utils.parse_time_string_to_ms(frame_size)
        self.stride = string_utils.parse_time_string_to_ms(stride)
        self.start = string_utils.parse_time_string_to_ms(start)
        self.end = string_utils.parse_time_string_to_ms(end)

        # Frame size 0 or None indicates that the whole session should be returned as one sample
        if self.frame_size == 0:
            warnings.warn(
                "Frame size should be bigger than zero. Returning whole session as sample."
            )

        # If the end time has not been set we initialize it with sys.maxsize
        if self.end is None or self.end == 0:
            self.end = sys.maxsize

        self.add_rest_class = add_rest_class
        self.fill_missing_data = fill_missing_data
        self.current_session = None

        # Data handler
        self._db_session_handler = SessionHandler(
            db_host, db_port, db_user, db_password
        )
        self._db_anno_handler = AnnotationHandler(
            db_host, db_port, db_user, db_password
        )
        self._db_stream_handler = StreamHandler(
            db_host, db_port, db_user, db_password, data_dir=data_dir
        )
        self._file_handler = FileHandler()

        self._iterable = self._yield_sample()

    def _init_data_from_description(self, data_desc: dict, dataset: str, session: str, header_only: bool = False) -> Data:
        """
        Initialize data from a data description.

        Args:
            data_desc (dict): Data description dictionary.
            dataset (str): Dataset name.
            session (str): Session name.
            header_only (bool): If true only the header information for all session data will be loaded.

        Returns:
            Data: Initialized data object.
        """
        src, type_, = data_desc["src"].split(":")
        if src == "db":
            if type_ == "anno":
                return self._db_anno_handler.load(
                    dataset=dataset,
                    session=session,
                    scheme=data_desc["scheme"],
                    annotator=data_desc["annotator"],
                    role=data_desc["role"],
                    header_only=header_only
                )
            elif type_ == "stream":
                try:
                    return self._db_stream_handler.load(
                        dataset=dataset,
                        session=session,
                        name=data_desc["name"],
                        role=data_desc["role"],
                        header_only=header_only
                    )
                except FileNotFoundError as e:
                    # Only raise file not found error if stream is requested as input
                    if not header_only:
                        raise e
                    # Create empty stream file with params
                    else:
                        # Todo differentiate types
                        empty_stream = Stream(None, -1, name=data_desc['name'], role=data_desc["role"], dataset=dataset, session=session)
                        return empty_stream


            else:
                raise ValueError(f"Unknown data type {type_} for data.")
        elif src == "file":
            return self._file_handler.load(fp=Path(data_desc["fp"]))
        else:
            raise ValueError(f"Unknown source type {src} for data.")


    def _data_description_to_string(self, data_desc: dict) -> str:
        """
        Convert data description to a string representation.

        Args:
            data_desc (dict): Data description dictionary.

        Returns:
            str: String representation of the data description.
        """

        id = data_desc.get("id")
        if id is not None:
            return id

        src, type_ = data_desc["src"].split(":")
        delim = "_"
        if src == "db":
            if type_ == "anno":
                return delim.join(
                    [data_desc["scheme"], data_desc["annotator"], data_desc["role"]]
                )
            elif type_ == "stream":
                return delim.join([data_desc["name"], data_desc["role"]])
            else:
                raise ValueError(f"Unknown data type {type_} for data.")
        elif src == "file":
            return delim.join([data_desc["fp"]])
        else:
            raise ValueError(f"Unknown source type {src} for data.")


    def init_session(self, session_name: str) -> Session:
        """
        Initialize a session.

        Args:
            session_name (str): Name of the session to initialize.

        Returns:
            Session: Initialized session object.
        """
        session = self._db_session_handler.load(self.dataset, session_name)

        """Opens all annotations and data readers"""
        input_data = {}
        output_data_templates = {}
        extra_data = {}

        # setting session data
        for data_desc in self.data:
            if data_desc.get('type') == 'input':
                data_initialized = self._init_data_from_description(
                    data_desc, self.dataset, session_name
                )
                data_id = self._data_description_to_string(data_desc)
                input_data[data_id] = data_initialized
            else:
                data_initialized = self._init_data_from_description(
                    data_desc, self.dataset, session_name, header_only=True
                )
                data_id = self._data_description_to_string(data_desc)
                if data_desc.get('type') == 'output':
                    output_data_templates[data_id] = data_initialized
                else:
                    extra_data[data_id] = data_initialized

        session.input_data = input_data
        session.extra_data = extra_data
        session.output_data_templates = output_data_templates

        # update session duration
        min_dur = session.duration if session.duration is not None else sys.maxsize
        for data_initialized in input_data.values():
            if isinstance(data_initialized, Stream):
                meta_data: StreamMetaData = data_initialized.meta_data
                if meta_data.duration is not None:
                    dur = meta_data.duration
                else:
                    dur = len(data_initialized.data) / meta_data.sample_rate * 1000
                if dur < min_dur:
                    min_dur = dur
        session.duration = min_dur

        if session.duration == sys.maxsize:
            raise ValueError(f"Unable to determine duration for session {session.name}")

        return session


    def _yield_sample(self) -> dict[str, np.ndarray]:
        """
        Yield examples.

        Yields:
            dict: Sampled data for the window.

        Example:
            {
                <scheme>_<annotator>_<role> : [[data1], [data2]... ],
                <file_path> : [[data1], [data2]... ]
            }
        """

        # Needed to sort the samples later and assure that the order is the same as in nova.
        #sample_counter = 1

        for session in self.sessions:
            # Init all data objects for the session and get necessary meta information
            self.current_session = self.init_session(session)

            # If frame size is zero or less we return the whole data from the whole session in one sample
            if self.frame_size <= 0:
                _frame_size = min(self.current_session.duration, self.end - self.start)
                _stride = _frame_size
            else:
                _frame_size = self.frame_size
                _stride = self.stride

            # Starting position of the first frame in seconds
            #cpos = max(self.left_context, self.start)
            cpos = 0

            # TODO account for stride and framesize being None
            # Generate samples for this session
            while cpos + self.stride <= min(
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
                # data_for_window = {
                #     k: v.sample_from_interval(window_start, window_end)
                #     for k, v in self.current_session.input_data.items()
                # }
                data_for_window = {}
                for k,v in self.current_session.input_data.items():

                    # TODO current_session duration is not the correct way to end right padding. We could have longer streams
                    start_ = max(0, window_start)
                    end_ = min(self.current_session.duration, window_end)
                    sample = v.sample_from_interval(start_, end_)

                    # TODO pad continuous annotations
                    # Apply padding
                    if isinstance(v, Stream):

                        # Don't pad anything but num_samples axis
                        sr = v.meta_data.sample_rate / 1000
                        left_pad = int((0-window_start) * sr) if window_start < 0 else 0
                        right_pad = int((window_end - self.current_session.duration) * sr) if window_end > self.current_session.duration else 0

                        if left_pad or right_pad:
                            lr_pad =  ((left_pad, right_pad), )
                            n_pad = tuple([(0,0)] * (len(sample.shape) -1 ))

                            # Num samples last dim
                            if isinstance(v, Audio):
                                pad = n_pad + lr_pad
                            # Num samples first dim
                            else:
                                pad = lr_pad + n_pad

                            sample = np.pad(sample, pad_width=pad, mode='edge')
                    data_for_window[k] = sample


                # pad
                #np.pad(data_for_window['audio'], pad_width=((0,0),(960,0)), mode='constant', constant_values=1)

                # Performing sanity checks
                garbage_detected = any(
                    [data_contains_garbage(d) for k, d in data_for_window.items()]
                )

                # Incrementing counter
                cpos += _stride
                #sample_counter += 1

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


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("../../../.env")
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")
    DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", None))

    dataset = "test"
    sessions = ["04_Oesterreich_test"]

    annotation = {
        "src": "db:anno",
        "scheme": "diarization",
        "type": "input",
        "id": "annotation",
        "annotator": "schildom",
        "role": "testrole2",
    }

    stream = {
        "src": "db:stream",
        "type": "input",
        "id": "featurestream",
        "role": "testrole",
        "name": "arousal.synchrony[testrole2]",
    }

    file = {
        "src": "file:stream",
        "type": "input",
        "id": "file",
        "fp": "/Users/dominikschiller/Work/local_nova_dir/test_files/new_test_video_25.mp4",
    }

    nova_iterator = NovaIterator(
        IP,
        PORT,
        USER,
        PASSWORD,
        dataset,
        DATA_DIR,
        sessions=sessions,
        data=[file],
        frame_size="1s",
        left_context="2s",
        right_context="2s",
        end="100s",
    )


    while next(nova_iterator):
        continue
    a = next(nova_iterator)
    breakpoint()
