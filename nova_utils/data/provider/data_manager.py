"""Data Storage Class for session specific data
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 25.10.2023
"""
from pathlib import Path

from nova_utils.data.annotation import FreeAnnotation, FreeAnnotationScheme
from nova_utils.data.data import Data
from nova_utils.data.handler import (
    file_handler,
    nova_db_handler,
    url_handler,
    request_handler,
)
from nova_utils.data.stream import Stream
from nova_utils.utils.request_utils import Source, DType, parse_src, dtype_from_desc, data_description_to_string


class SessionManager:
    """
    Class to aggregate and manage interrelated incoming and outgoing datastreams belonging to a single session (e.g multimodal data from the same recording).

    Attributes:
        input_data (dict, optional):  Annotation or Stream data that can be processed by a module.
        dataset (str, optional): The dataset or category the session belongs to.
        data_description (list[dict[str, str]], optional): List of data descriptions. Defaults to None. The dictionary should have the following mandatory fields:
        source_context (dict[str, dict]) : Dict of parameters that are need to interact with a source. E.g. database credentials or data directories. Must match constructor arguments of the respective data handler.
        session (str, optional): The name or title of the session.
        duration (int, optional): The duration of the session in minutes.
        location (str, optional): The location or venue of the session.
        language (str, optional): The language used in the session.
        date (datetime, optional): The date and time of the session.
        is_valid (bool, optional): Whether the session is considered valid.
        extra_data (dict, optional): Additional stream or annotation information for the session. Might only contain meta information.
        output_data_templates (dict, optional): Empty data objects that contain meta information for outputs. Modules can fill the data here.

    Args:
        dataset (str, optional): The dataset or category the session belongs to. Must match NovaDB entries if 'db' is
        data_description (list[dict[str, str]], optional): List of data descriptions. Defaults to None. The dictionary should have the following mandatory fields:
            ``"id"``:
                Unique id to map the data to a given input / output.
            ``"name"``:
                Output name for streams
            ``"type"``:
                IO type of the data. Either "input" or "output"
            ``"src"``
                The source and datatype to load the data from separated by ':' . Source is of typ Source.value and datatype of type DType.value
                E.g. 'db:anno'

            In addition, each entry should provide the information that is need to identify the exact input and output targets.
            Must match the input parameters of the respective data handlers save and load function (dataset and session are already specified as properties).
            E.g. to load an annotation from NovaDB we use the data.handler.nova_db_handler.AnnotationHandler we need the following additional fields.
                ``"scheme"``
                    The scheme name of the annotations to load. Only necessary when loading annotations from the database.
                ``"annotator"``
                    The annotator of the annotations to load. Only necessary when loading annotations from the database.
                ``"role"``
                    The role to which the data belongs. Only necessary when accessing data from the database.
            To load a stream file from disk using data.handler.file_handler.FileHandler we only need a filepath
            ``"uri"``
                The filepath from which to load the data from. Only necessary when loading files from disk.

        session (str, optional): The name or title of the session.
        source_context (dict[str, dict]) : List of parameters that are need to interact with a source. E.g. database credentials or data directories. Must match constructor arguments of the respective data handler.
        duration (int, optional): The duration of the session in milliseconds.
        location (str, optional): The location or venue of the session.
        language (str, optional): The language used in the session.
        date (datetime, optional): The date and time of the session.
        is_valid (bool, optional): Whether the session is considered valid.
        input_data (dict, optional): Annotation or Stream data that can be processed by a module.
        extra_data (dict, optional): Additional stream or annotation information for the session. Might only contain meta information.
        output_data_templates (dict, optional): Empty data objects that contain meta information for outputs. Modules can fill the data here.
    """

    def __init__(
        self,
        dataset: str = None,
        data_description: list[dict[str, str]] = None,
        session: str = None,
        source_context: dict[str, dict] = None,
        input_data: dict = None,
        extra_data: dict = None,
        output_data_templates: dict = None,
    ):
        self.dataset = dataset
        self.data_description = data_description
        self.session = session
        self.input_data = {} if input_data is None else input_data
        self.extra_data = {} if extra_data is None else extra_data
        self.output_data_templates = (
            {} if output_data_templates is None else output_data_templates
        )

        self.source_context = {}
        for src, context in source_context.items():
            src_ = Source(src)
            self.add_source_context(src_, context)

    def add_source_context(self, source: Source, context: dict):
        """Add all parameters that are necessary to initialize source specific data handler for reading and writing data objects."""
        self.source_context[source] = context


    def _update_data_description(self, data_description=None):
        if data_description is not None:
            self.data_description = data_description
        return self.data_description


    def load(self, data_description=None):
        """
        Args:
            data_description (list[dict[str, str]], optional): List of data descriptions. Defaults to None. The dictionary should have the following mandatory fields:
            ``"id"``:
                Unique id to map the data to a given input / output.
            ``"name"``:
                Output name for streams
            ``"type"``:
                IO type of the data. Either "input" or "output"
            ``"src"``
                The source and datatype to load the data from separated by ':' . Source is of typ Source.value and datatype of type DType.value
                E.g. 'db:anno'

            In addition, each entry should provide the information that is need to identify the exact input and output targets.
            Must match the input parameters of the respective data handlers save and load function (dataset and session are already specified as properties).
            E.g. to load an annotation from NovaDB we use the data.handler.nova_db_handler.AnnotationHandler we need the following additional fields.
                ``"scheme"``
                    The scheme name of the annotations to load. Only necessary when loading annotations from the database.
                ``"annotator"``
                    The annotator of the annotations to load. Only necessary when loading annotations from the database.
                ``"role"``
                    The role to which the data belongs. Only necessary when accessing data from the database.
            To load a stream file from disk using data.handler.file_handler.FileHandler we only need a filepath
            ``"fp"``
                The filepath from which to load the data from. Only necessary when loading files from disk.

        Returns:

        """

        data_description = self._update_data_description(data_description)
        if data_description is None:
            raise ValueError(
                "Data description is empty. Either pass a data description to load() or set it as class attribute."
            )

        for desc in data_description:
            src, dtype, dtype_specific = parse_src(desc)

            header_only = False
            if desc.get("type") == "input":
                io_dst = self.input_data
            elif desc.get("type") == "output":
                io_dst = self.output_data_templates
                header_only = True
            else:
                io_dst = self.extra_data

            data_id = data_description_to_string(desc)
            data = None

            if src in [Source.DB] and not src in self.source_context.keys():
                raise ValueError(
                    f"Missing context information source {src}. Call add_source_context() first."
                )
            try:
                # DATABASE
                if src == Source.DB:
                    ctx = self.source_context[src]
                    if dtype == DType.ANNO:
                        handler = nova_db_handler.AnnotationHandler(**ctx)
                        data = handler.load(
                            dataset=self.dataset,
                            session=self.session,
                            scheme=desc["scheme"],
                            annotator=desc["annotator"],
                            role=desc["role"],
                            header_only=header_only,
                        )
                    elif dtype == DType.STREAM:
                        handler = nova_db_handler.StreamHandler(**ctx)
                        data = handler.load(
                            dataset=self.dataset,
                            session=self.session,
                            name=desc["name"],
                            role=desc["role"],
                            header_only=header_only,
                        )
                # FILE
                elif src == Source.FILE:
                    # Need to set the file handler specifically because we don't know the scheme
                    if dtype == DType.ANNO:
                        if dtype_specific is None or dtype_specific == 'free':
                            data = FreeAnnotation(scheme=FreeAnnotationScheme(name='generic'), data=None)
                        else:
                            raise ValueError(f"Can\'t create template for {desc} because no scheme information is available.")
                    # Automatic file handler detection
                    else:
                        handler = file_handler.FileHandler()
                        data = handler.load(fp=Path(desc["uri"]), header_only=header_only)
                # URL
                elif src == Source.URL:
                    handler = url_handler.URLHandler()
                    data = handler.load(uri=Path(desc["uri"]))
                # REQUEST
                elif src == Source.REQUEST:
                    target_dtype = dtype_from_desc(desc)
                    handler = request_handler.RequestHandler()
                    data = handler.load(data=desc.get("data"), dtype=target_dtype, header_only=header_only)

            except FileNotFoundError as e:
                # Only raise file not found error if stream is requested as input
                if not header_only:
                    raise e
                # Create empty data objects with known params
                else:
                    if dtype == DType.STREAM:
                        # Todo differentiate types
                        data = Stream(
                            None,
                            -1,
                            name=desc.get("name"),
                            role=desc.get("role"),
                            dataset=self.dataset,
                            session=self.session,
                        )

                    else:
                        # Todo Handle other cases where no header might be loaded
                        data = Data()

            io_dst[data_id] = data

    def save(self, data_description=None):
        """
        Args:
          data_description (list[dict[str, str]], optional): List of data descriptions. Defaults to None. The dictionary should have the following mandatory fields:
          ``"id"``:
              Unique id to map the data to a given input / output.
          ``"name"``:
              Output name for streams
          ``"type"``:
              IO type of the data. Either "input" or "output"
          ``"src"``
              The source and datatype to load the data from separated by ':' . Source is of typ Source.value and datatype of type DType.value
              E.g. 'db:anno'

          In addition, each entry should provide the information that is need to identify the exact input and output targets.
          Must match the input parameters of the respective data handlers save and load function (dataset and session are already specified as properties).
          E.g. to load an annotation from NovaDB we use the data.handler.nova_db_handler.AnnotationHandler we need the following additional fields.
              ``"scheme"``
                  The scheme name of the annotations to load. Only necessary when loading annotations from the database.
              ``"annotator"``
                  The annotator of the annotations to load. Only necessary when loading annotations from the database.
              ``"role"``
                  The role to which the data belongs. Only necessary when accessing data from the database.
          To load a stream file from disk using data.handler.file_handler.FileHandler we only need a filepath
          ``"fp"``
              The filepath from which to load the data from. Only necessary when loading files from disk.

        Returns:
        """

        data_description = self._update_data_description(data_description)
        if data_description is None:
            raise ValueError(
                "Data description is empty. Either pass a data description to save() or set it as class attribute."
            )

        for desc in data_description:
            src, dtype_specific, dtype = parse_src(desc)

            if not desc.get("type") == "output":
                continue

            data_id = data_description_to_string(desc)

            if src in [Source.DB] and not src in self.source_context.keys():
                raise ValueError(
                    f"Missing context information source {src}. Call add_source_context() first."
                )

            success = False
            data = self.output_data_templates[data_id]
            if src == Source.DB:
                ctx = self.source_context[src]
                if dtype == DType.ANNO:
                    handler = nova_db_handler.AnnotationHandler(**ctx)
                    success = handler.save(annotation=data)
                elif dtype == DType.STREAM:
                    handler = nova_db_handler.StreamHandler(**ctx)
                    success = handler.save(stream=data)
            elif src == Source.FILE:
                handler = file_handler.FileHandler()
                success = handler.save(data=data, fp=Path(desc["uri"]))
            elif src == Source.URL:
                raise NotImplementedError
            elif src == Source.REQUEST:
                rq = self.source_context.get(Source.REQUEST)
                shared_dir = rq.get('shared_dir')
                job_id = rq.get('job_id')
                handler = request_handler.RequestHandler()
                handler.save(data=data, shared_dir=shared_dir, job_id=job_id, dataset=self.dataset, session=self.session)

            return success


class DatasetManager:
    def __init__(
        self, dataset, data_description, source_context, session_names: list = None
    ):
        self.dataset = dataset
        self.data_description = data_description
        self.session_names = session_names
        self.source_ctx = source_context
        self.sessions = {}
        self._init_sessions()

    def _init_sessions(self):
        if self.session_names is not None:
            for session in self.session_names:
                sm = SessionManager(
                    self.dataset, self.data_description, session, self.source_ctx
                )
                self.sessions[session] = {"manager": sm}

    def load_session(self, session_name):
        self.sessions[session_name]["manager"].load(self.data_description)

    def save_session(self, session_name):
        self.sessions[session_name]["manager"].save(self.data_description)

    def load(self):
        for session in self.session_names:
            self.load_session(session)

    def save(self):
        for session in self.session_names:
            self.save_session(session)


class NovaDatasetManager(DatasetManager):
    def _init_sessions(self):
        sh = nova_db_handler.SessionHandler(**self.source_ctx["db"])
        sessions = sh.load(self.dataset, self.session_names)
        for session_info in sessions:
            sm = SessionManager(
                self.dataset, self.data_description, session_info.name, self.source_ctx
            )
            self.sessions[session_info.name] = {"manager": sm, "info": session_info}


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv("../../../.env")
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")
    DATA_DIR = Path(os.getenv("NOVA_DATA_DIR", None))

    dataset = "test"
    sessions = ["01_AffWild2_video1"]

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
        "uri": "/Users/dominikschiller/Work/local_nova_dir/test_files/new_test_video_25.mp4",
    }

    ctx = {
        "db": {
            "db_host": IP,
            "db_port": PORT,
            "db_user": USER,
            "db_password": PASSWORD,
            "data_dir": DATA_DIR,
        },
    }

    annotation_out = {
        "src": "file:anno",
        "type": "output",
        "id": "annotation_out",
        "uri": "./test_output.annotation",
    }

    # data_aggregator_in = SessionManager(
    #     dataset=dataset,
    #     session=sessions[0],
    #     source_context=ctx,
    # )

    data_aggregator_in = NovaDatasetManager(
        dataset=dataset, data_description=[annotation], **ctx["db"]
    )

    data_aggregator_in.load(data_description=[annotation, annotation_out])
    data_aggregator_in.output_data_templates[
        "annotation_out"
    ] = data_aggregator_in.input_data["annotation"]
    data_aggregator_in.save(data_description=[annotation, annotation_out])

    breakpoint()
