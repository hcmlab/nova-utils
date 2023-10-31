""" Module for handling MongoDB data operations related to annotations and streams.

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    18.8.2023
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.results import InsertOneResult, UpdateResult

from nova_utils.data.annotation import (
    Annotation,
    DiscreteAnnotation,
    DiscreteAnnotationScheme,
    ContinuousAnnotation,
    ContinuousAnnotationScheme,
    FreeAnnotation,
    FreeAnnotationScheme,
)
from nova_utils.data.handler.file_handler import FileHandler
from nova_utils.data.handler.ihandler import IHandler
from nova_utils.data.stream import Stream, SSIStream, StreamMetaData
from nova_utils.utils.anno_utils import (
    convert_ssi_to_label_dtype,
    convert_label_to_ssi_dtype,
)
from nova_utils.utils.type_definitions import SSILabelDType, SchemeType

ANNOTATOR_COLLECTION = "Annotators"
SCHEME_COLLECTION = "Schemes"
STREAM_COLLECTION = "Streams"
ROLE_COLLECTION = "Roles"
ANNOTATION_COLLECTION = "Annotations"
SESSION_COLLECTION = "Sessions"
ANNOTATION_DATA_COLLECTION = "AnnotationData"

# METADATA
class NovaDBMetaData:
    """
    Metadata for MongoDB connection.

    Attributes:
        ip (str, optional): IP address of the MongoDB server.
        port (int, optional): Port number of the MongoDB server.
        user (str, optional): Username for authentication.
        dataset (str, optional): Name of the dataset.
    """

    def __init__(
        self, ip: str = None, port: int = None, user: str = None, dataset: str = None
    ):
        self.ip = ip
        self.port = port
        self.user = user
        self.dataset = dataset


class NovaDBAnnotationMetaData(NovaDBMetaData):
    """
    Metadata for MongoDB annotations.

    Attributes:
        is_locked (bool, optional): Indicates if the annotation is locked.
        is_finished (bool, optional): Indicates if the annotation is finished.
        last_update (bool, optional): Timestamp of the last update.
        annotation_document_id (ObjectId, optional): ID of the annotation document.
        data_document_id (ObjectId, optional): ID of the associated data document.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        is_locked: bool = None,
        is_finished: bool = None,
        last_update: bool = None,
        annotation_document_id: ObjectId = None,
        data_document_id: ObjectId = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_locked = is_locked
        self.is_finished = is_finished
        self.last_update = last_update
        self.annotation_document_id = annotation_document_id
        self.data_document_id = data_document_id


class NovaDBStreamMetaData(NovaDBMetaData):
    """
    Metadata for MongoDB streams.

    Attributes:
        name (str, optional): Name of the stream.
        dim_labels (dict, optional): Dimension labels of the stream.
        file_ext (str, optional): File extension of the stream data.
        is_valid (bool, optional): Indicates if the stream data is valid.
        stream_document_id (ObjectId, optional): ID of the stream document.
        db_sample_rate (float, optional): Sample rate of the stream data in the database.
        type (str, optional): Type of the stream data.
         **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        name: str = None,
        dim_labels: dict = None,
        file_ext: str = None,
        is_valid: bool = None,
        stream_document_id: ObjectId = None,
        sr: float = None,
        type: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = name
        self.dim_labels = dim_labels
        self.file_ext = file_ext
        self.is_valid = is_valid
        self.stream_document_id = stream_document_id
        self.db_sample_rate = sr
        self.type = type


# DATA
class NovaDBHandler:
    """
    Base class for handling MongoDB connections.

     Args:
        db_host (str, optional): IP address of the MongoDB server.
        db_port (int, optional): Port number of the MongoDB server.
        db_user (str, optional): Username for authentication.
        db_password (str, optional): Password for authentication.
        data_dir (str, optional): Datadir to load streams from

    Attributes:
        data_dir (Path, optional): Path to the data directory for stream files
    """

    def __init__(
        self,
        db_host: str = None,
        db_port: int = None,
        db_user: str = None,
        db_password: str = None,
        data_dir: str = None
    ):
        self._client = None
        self._ip = None
        self._port = None
        self._user = None
        self.data_dir = None if data_dir is None else Path(data_dir)
        if db_host and db_port and db_user and db_password:
            self.connect(db_host, db_port, db_user, db_password)

    def connect(
        self, db_host: str = None, db_port: int = None, db_user: str = None, db_password: str = None
    ):
        """
        Connects to the MongoDB server.

        Args:
            db_host (str): IP address of the MongoDB server.
            db_port (int): Port number of the MongoDB server.
            db_user (str): Username for authentication.
            db_password (str): Password for authentication.
        """
        self._client = MongoClient(host=db_host, port=db_port, username=db_user, password=db_password)
        self._ip = db_host
        self._port = db_port
        self._user = db_user

    @property
    def client(self) -> MongoClient:
        """
        Returns the MongoDB client instance.
        """
        return self._client

class NovaSession:
    """
    Class to stores all information belonging to a specific session during processing

    Attributes:
        input_data (dict, optional):  Annotation or Stream data that can be processed by a module.
        dataset (str, optional): The dataset or category the session belongs to.
        name (str, optional): The name or title of the session.
        duration (int, optional): The duration of the session in minutes.
        location (str, optional): The location or venue of the session.
        language (str, optional): The language used in the session.
        date (datetime, optional): The date and time of the session.


    Args:
        dataset (str, optional): The dataset or category the session belongs to.
        name (str, optional): The name or title of the session.
        duration (int, optional): The duration of the session in milliseconds.
        location (str, optional): The location or venue of the session.
        language (str, optional): The language used in the session.
        date (datetime, optional): The date and time of the session.
        is_valid (bool, optional): Whether the session is considered valid.
    """
    def __init__(self, input_data: dict = None, dataset: str = None, name: str = None, duration: int = None, location: str = None, language: str = None, date: datetime = None, is_valid: bool = True, extra_data: dict = None, output_data_templates: dict = None):
        self.input_data = input_data
        self.dataset = dataset
        self.name = name
        self.duration = duration
        self.location = location
        self.language = language
        self.date = date

class SessionHandler(NovaDBHandler):
    """
    Handler for loading session data from a MongoDB database.

    This class provides methods to load session information from a MongoDB
    collection and returns a Session object.

    Args:
         (Inherited args from MongoHandler)

    Methods:
        load(dataset: str, session: str) -> Session:
            Load session data from the specified dataset and session name.

    Attributes:
        (Inherited attributes from MongoHandler)

    """

    def load(self, dataset: str, session: Union[str, list, None] = None) -> list[NovaSession]:
        """
        Load session data from the specified dataset and session name.

        Args:
            dataset (str): The dataset name as specified in the mongo database
            session (str, list, None): The session name as specified in the mongo database. Can be also a list of session names. If no session name is provided, all sessions are loaded instead.

        Returns:
            NovaSession: A Session object containing loaded session information.
            If the session does not exist, an empty Session object is returned.
        """
        if isinstance(session, str):
            session = [session]

        ret = []
        result = self.client[dataset][SESSION_COLLECTION].find({})

        for s in result:
            if session is not None and s['name'] not in session:
                continue
            # get duration of session in milliseconds
            dur_ms = s.get("duration")
            if dur_ms == 0:
                dur_ms = None
            else:
                dur_ms *= 1000

            ret.append(
                NovaSession(
                    dataset=dataset,
                    name=s["name"],
                    location=s["location"],
                    language=s["language"],
                    date=s["date"],
                    duration=dur_ms,
                    is_valid=s["isValid"],
                )
            )

        return ret



class AnnotationHandler(IHandler, NovaDBHandler):
    """
    Class for handling download of annotation data from Mongo db.
    """

    def _load_annotation(
        self,
        dataset: str,
        session: str,
        annotator: str,
        role: str,
        scheme: str,
        project: dict = None,
    ) -> dict:
        """
        Load annotation data from MongoDB.

        Args:
            dataset (str): Name of the dataset.
            session (str): Name of the session.
            annotator (str): Name of the annotator.
            role (str): Name of the role.
            scheme (str): Name of the annotation scheme.
            project (dict, optional): Projection for MongoDB query to filter attributes. Defaults to None.

        Returns:
            dict: Loaded annotation data.
        """
        pipeline = [
            {
                "$lookup": {
                    "from": SESSION_COLLECTION,
                    "localField": "session_id",
                    "foreignField": "_id",
                    "as": "session",
                }
            },
            {
                "$lookup": {
                    "from": ANNOTATOR_COLLECTION,
                    "localField": "annotator_id",
                    "foreignField": "_id",
                    "as": "annotator",
                }
            },
            {
                "$lookup": {
                    "from": ROLE_COLLECTION,
                    "localField": "role_id",
                    "foreignField": "_id",
                    "as": "role",
                }
            },
            {
                "$lookup": {
                    "from": SCHEME_COLLECTION,
                    "localField": "scheme_id",
                    "foreignField": "_id",
                    "as": "scheme",
                }
            },
            {
                "$match": {
                    "$and": [
                        {"role.name": role},
                        {"session.name": session},
                        {"annotator.name": annotator},
                        {"scheme.name": scheme},
                    ]
                }
            },
            {
                "$lookup": {
                    "from": "AnnotationData",
                    "localField": "data_id",
                    "foreignField": "_id",
                    "as": "data",
                }
            },
        ]

        # append projection
        if project:
            pipeline.append({"$project": project})

        result = list(self.client[dataset][ANNOTATION_COLLECTION].aggregate(pipeline))
        if not result:
            return {}
        return result[0]

    def _load_scheme(self, dataset:str, scheme:str) -> dict:
        result = self.client[dataset][SCHEME_COLLECTION].find_one({"name": scheme})
        if not result:
            return {}
        return result

    def _update_annotation(
        self,
        dataset: str,
        annotation_id: ObjectId,
        annotation_data_id: ObjectId,
        annotation_data: list[dict],
        is_finished: bool,
        is_locked: bool,
    ) -> UpdateResult:
        """
        Updates existing annotation the Mongo database

        Args:
            dataset (str): Name of the dataset.
            annotation_id (ObjectId): ObjectId of the annotation in the database
            annotation_data_id (ObjectId): ObjectId of the corresponding annotation data object in the database
            annotation_data (list[dict]): List of dictionaries containing the annotation data. Each dictionary represents one sample. Keys must match the annotation types.
            is_finished (bool): Whether the annotation has already been fully completed or not
            is_locked (bool): Whether the annotation should be locked and can therefore not be overwritten anymore.

        Returns:
            UpdateResult: The success status of the update operation
        """
        update_query_annotation = {
            "$set": {
                "date": datetime.now(),
                "isFinished": is_finished,
                "isLocked": is_locked,
            }
        }
        update_query_annotation_data = {"$set": {"labels": annotation_data}}
        success = self.client[dataset][ANNOTATION_COLLECTION].update_one(
            {"_id": annotation_id}, update_query_annotation
        )

        if not success.acknowledged:
            return success

        success = self.client[dataset][ANNOTATION_DATA_COLLECTION].update_one(
            {"_id": annotation_data_id}, update_query_annotation_data
        )

        return success

    def _insert_annotation_data(self, dataset: str, data: list) -> InsertOneResult:
        """
        Insert annotation data into the MongoDB database.

        Args:
            dataset (str): Name of the dataset.
            data (list): List of annotation data to be inserted.

        Returns:
            InsertOneResult: The result of the insertion operation.
        """
        annotation_data = {"labels": data}

        success = self.client[dataset][ANNOTATION_DATA_COLLECTION].insert_one(
            annotation_data
        )
        return success

    def _insert_annotation(
        self,
        dataset: str,
        session_id: ObjectId,
        annotator_id: ObjectId,
        scheme_id: ObjectId,
        role_id: ObjectId,
        data: list,
        is_finished: bool,
        is_locked: bool,
    ):
        """
        Insert annotation and associated annotation data into the MongoDB database.

        Args:
            dataset (str): Name of the dataset.
            session_id (ObjectId): ID of the associated session.
            annotator_id (ObjectId): ID of the annotator.
            scheme_id (ObjectId): ID of the annotation scheme.
            role_id (ObjectId): ID of the role.
            data (list): List of annotation data to be inserted.
            is_finished (bool): Indicates if the annotation is finished.
            is_locked (bool): Indicates if the annotation is locked.

        Returns:
            InsertOneResult: The result of the insertion operation.
        """
        # insert annotation data first
        success = self._insert_annotation_data(dataset, data)
        if not success.acknowledged:
            return success
        else:
            data_id = success.inserted_id

        # insert annotation object
        annotation_document = {
            "session_id": session_id,
            "annotator_id": annotator_id,
            "scheme_id": scheme_id,
            "role_id": role_id,
            "date": datetime.now(),
            "isFinished": is_finished,
            "isLocked": is_locked,
            "data_id": data_id,
        }
        success = self.client[dataset][ANNOTATION_COLLECTION].insert_one(
            annotation_document
        )

        # if the annotation could not be created we delete the annotation data as well
        if not success.acknowledged:
            success = self.client[dataset][ANNOTATION_DATA_COLLECTION].delete_one(
                {"_id": data_id}
            )

        return success


    def load(
        self, dataset: str, scheme: str, session: str, annotator: str, role: str, header_only: bool = False
    ) -> Annotation:
        """
        Load annotation data from MongoDB and create an Annotation object.

        Args:
            dataset (str): Name of the dataset.
            scheme (str): Name of the annotation scheme.
            session (str): Name of the session.
            annotator (str): Name of the annotator.
            role (str): Name of the role.
            header_only (bool): If true only the annotation header will be loaded.


        Returns:
            Annotation: An Annotation object loaded from the database.

        Raises:
            FileNotFoundError: If the requested annotation data is not found in the database.
            TypeError: If the scheme type is unknown.
        """
        # load annotation from mongo db
        if header_only:
            scheme_doc = self._load_scheme(dataset, scheme)

            if not scheme_doc:
                raise FileNotFoundError(
                    f"Scheme not found dataset: {dataset} scheme: {scheme}"
                )

        else:
            anno_doc = self._load_annotation(dataset, session, annotator, role, scheme)

            if not anno_doc:
                raise FileNotFoundError(
                    f"Annotation not found dataset: {dataset} session: {session} annotator: {annotator} role: {role} scheme: {scheme}"
                )

            (anno_data_doc,) = anno_doc["data"]

            # build annotation object
            (scheme_doc,) = anno_doc["scheme"]

        scheme_type = scheme_doc["type"]
        anno_data = None
        anno_duration = 0

        # discrete scheme
        if scheme_type == SchemeType.DISCRETE.name:
            scheme_classes = {l["id"]: l["name"] for l in scheme_doc["labels"]}

            if not header_only:
                anno_data = np.array(
                    [
                        (x["from"], x["to"], x["id"], x["conf"])
                        for x in anno_data_doc["labels"]
                    ],
                    dtype=SSILabelDType.DISCRETE.value,
                )
                anno_data = convert_ssi_to_label_dtype(anno_data, SchemeType.DISCRETE)
                anno_duration = anno_data[-1]["to"] if anno_data.size != 0 else 0

            anno_scheme = DiscreteAnnotationScheme(name=scheme, classes=scheme_classes)
            annotation = DiscreteAnnotation(
                role=role,
                session=session,
                dataset=dataset,
                data=anno_data,
                scheme=anno_scheme,
                annotator=annotator,
                duration=anno_duration,
            )

        # continuous scheme
        elif scheme_type == SchemeType.CONTINUOUS.name:
            sr = scheme_doc["sr"]
            min_val = scheme_doc["min"]
            max_val = scheme_doc["max"]

            if not header_only:
                anno_data = np.array(
                    [(x["score"], x["conf"]) for x in anno_data_doc["labels"]],
                    dtype=SSILabelDType.CONTINUOUS.value,
                )
                anno_data = convert_ssi_to_label_dtype(anno_data, SchemeType.CONTINUOUS)
                anno_duration = len(anno_data_doc["labels"]) / sr

            anno_scheme = ContinuousAnnotationScheme(
                name=scheme, sample_rate=sr, min_val=min_val, max_val=max_val
            )
            annotation = ContinuousAnnotation(
                role=role,
                session=session,
                dataset=dataset,
                data=anno_data,
                scheme=anno_scheme,
                annotator=annotator,
                duration=anno_duration,
            )

            # free scheme

        # free scheme
        elif scheme_type == SchemeType.FREE.name:

            if not header_only:
                anno_data = np.array(
                    [
                        (x["from"], x["to"], x["name"], x["conf"])
                        for x in anno_data_doc["labels"]
                    ],
                    dtype=SSILabelDType.FREE.value,
                )

                anno_data = convert_ssi_to_label_dtype(anno_data, SchemeType.FREE)
                anno_duration = anno_data[-1]["to"] if anno_data.size != 0 else 0

            anno_scheme = FreeAnnotationScheme(name=scheme)
            annotation = FreeAnnotation(
                role=role,
                session=session,
                dataset=dataset,
                data=anno_data,
                scheme=anno_scheme,
                annotator=annotator,
                duration=anno_duration,
            )
        else:
            raise TypeError(f"Unknown scheme type {scheme_type}")

        # setting meta data
        if header_only:
            handler_meta_data = NovaDBAnnotationMetaData(
                ip=self._ip,
                port=self._port,
                user=self._user,
                dataset=dataset
            )
        else:
            handler_meta_data = NovaDBAnnotationMetaData(
                ip=self._ip,
                port=self._port,
                user=self._user,
                dataset=dataset,
                is_locked=anno_doc.get("isLocked"),
                is_finished=anno_doc.get("isFinished"),
                annotation_document_id=anno_doc.get("_id"),
                data_document_id=anno_doc.get("data_id"),
                last_update=anno_doc.get("date"),
            )
        annotation.meta_data.expand(handler_meta_data)

        return annotation

    def save(
        self,
        annotation: Annotation,
        dataset: str = None,
        session: str = None,
        annotator: str = None,
        role: str = None,
        is_finished: bool = False,
        is_locked: bool = False,
        overwrite: bool = False,
    ):
        """
        Save an Annotation object to the MongoDB database.

        Args:
            annotation (Annotation): The Annotation object to be saved.
            dataset (str, optional): Name of the dataset. Overwrites the respective attribute from annotation.meta_data if set. Defaults to None.
            session (str, optional): Name of the session. Overwrites the respective attribute from annotation.meta_data if set. Defaults to None.
            annotator (str, optional): Name of the annotator. Overwrites the respective attribute from annotation.meta_data if set. Defaults to None.
            role (str, optional): Name of the role. Overwrites the respective attribute from annotation.meta_data if set. Defaults to None.
            is_finished (bool, optional): Indicates if the annotation is finished. Defaults to False.
            is_locked (bool, optional): Indicates if the annotation is locked. Defaults to False.
            overwrite (bool, optional): If True, overwrite an existing annotation. Defaults to False.

        Returns:
            UpdateResult: The result of the update operation.

        Raises:
            FileExistError: If annotation exists and is locked or annotation exists and overwrite is set to false
        """
        # overwrite default values
        dataset = dataset if not dataset is None else annotation.meta_data.dataset
        session = session if not session is None else annotation.meta_data.session
        annotator = annotator if not annotator is None else annotation.meta_data.annotator
        role = role if not role is None else annotation.meta_data.role
        scheme = annotation.annotation_scheme.name

        if isinstance(annotation.data, np.ndarray) and not annotation.data.size == 0:
            anno_data = convert_label_to_ssi_dtype(
                annotation.data, annotation.annotation_scheme.scheme_type
            )

            # TODO check for none values
            anno_data = [
                dict(zip(annotation.annotation_scheme.label_dtype.names, ad.item()))
                for ad in anno_data
            ]
        else:
            anno_data = []

        # load annotation to check if an annotation for the provided criteria already exists in the database
        anno_doc = self._load_annotation(
            dataset,
            session,
            annotator,
            role,
            scheme,
            project={"_id": 1, "isLocked": 1, "data_id": 1},
        )

        # update existing annotations
        if anno_doc:
            if anno_doc["isLocked"]:
                raise FileExistsError(
                    f"Can't overwrite locked annotation dataset: {dataset} session: {session} annotator: {annotator} role: {role} scheme: {scheme}. Because annotation is locked."
                )
            elif not overwrite:
                raise FileExistsError(
                    f"Can't overwrite annotation dataset: {dataset} session: {session} annotator: {annotator} role: {role} scheme: {scheme}. Because overwrite is disabled."
                )
            else:
                warnings.warn(
                    f"Overwriting existing annotation dataset: {dataset} session: {session} annotator: {annotator} role: {role} scheme: {scheme}"
                )

                success = self._update_annotation(
                    dataset=dataset,
                    annotation_id=anno_doc["_id"],
                    annotation_data_id=anno_doc["data_id"],
                    annotation_data=anno_data,
                    is_finished=is_finished,
                    is_locked=is_locked,
                )

        # add new annotation
        else:
            scheme_id = self.client[dataset][SCHEME_COLLECTION].find_one(
                {"name": scheme}
            )["_id"]
            session_id = self.client[dataset][SESSION_COLLECTION].find_one(
                {"name": session}
            )["_id"]
            role_id = self.client[dataset][ROLE_COLLECTION].find_one({"name": role})[
                "_id"
            ]
            annotator_id = self.client[dataset][ANNOTATOR_COLLECTION].find_one(
                {"name": annotator}
            )["_id"]
            success = self._insert_annotation(
                dataset=dataset,
                scheme_id=scheme_id,
                session_id=session_id,
                annotator_id=annotator_id,
                role_id=role_id,
                data=anno_data,
                is_finished=is_finished,
                is_locked=is_locked,
            )

        return success
        # TODO success error handling


class StreamHandler(IHandler, NovaDBHandler):
    """
    Class for handling download and upload of stream data from MongoDB.
    """


    def _load_stream(
        self,
        dataset: str,
        stream_name: str,
    ) -> dict:
        """
        Load stream data from MongoDB.

        Args:
            dataset (str): Name of the dataset.
            stream_name (str): Name of the stream.

        Returns:
            dict: Loaded stream data.
        """
        result = self.client[dataset][STREAM_COLLECTION].find_one({"name": stream_name})
        if not result:
            return {}
        return result

    def load(self, dataset: str, session: str, role: str, name: str, header_only: bool = False) -> Stream:
        """
        Load a Stream object from MongoDB and create a Stream instance.

        Args:
            dataset (str): Name of the dataset.
            session (str): Name of the session.
            role (str): Name of the role.
            name (str): Name of the stream.
            header_only (bool): If true only the stream header will be loaded.

        Returns:
            Stream: A Stream object loaded from the database.

        Raises:
            ValueError: If the requested stream is not found for the given dataset.
            FileNotFoundError: If the data directory is not set or the file is not found on disc.
        """
        result = self._load_stream(dataset=dataset, stream_name=name)
        if not result:
            raise FileNotFoundError(f"No stream {name} found for dataset {dataset}")
        if not self.data_dir:
            raise FileNotFoundError("Data directory was not set. Can't access files")

        file_path = Path(
            self.data_dir
            / dataset
            / session
            / (role + "." + result["name"] + "." + result["fileExt"])
        )

        if not file_path.is_file():
            raise FileNotFoundError(f"No such file {file_path}")

        # data
        data = FileHandler().load(file_path, header_only = header_only)
        assert isinstance(data, Stream)

        # meta data
        data.meta_data.role = role
        data.meta_data.name = name
        data.meta_data.session = session

        handler_meta_data = NovaDBStreamMetaData(
            ip=self._ip,
            port=self._port,
            user=self._user,
            name=result.get("name"),
            dim_labels=result.get("dimLabels"),
            file_ext=result.get("fileExt"),
            is_valid=result.get("isValid"),
            stream_document_id=result.get("_id"),
            sr=result.get("sr"),
            type=result.get("type"),
            dataset=dataset
        )
        data.meta_data.expand(handler_meta_data)

        return data

    def save(
        self,
        stream: Stream,
        dataset: str = None,
        session: str = None,
        role: str = None,
        name: str = None,
        media_type: str = None,
        file_ext: str = None,
        dim_labels: list = None,
        is_valid: bool = True,
    ):
        """
        Save a Stream object to the MongoDB database and store associated file.

        Args:
            stream (Stream): The Stream object to be saved.
            dataset (str): Name of the dataset. Overwrites the respective attribute from stream.meta_data if set. Defaults to None.
            session (str): Name of the session. Overwrites the respective attribute from stream.meta_data if set. Defaults to None.
            role (str): Name of the role. Overwrites the respective attribute from stream.meta_data if set. Defaults to None.
            name (str): Name of the stream. Overwrites the respective attribute from stream.meta_data if set. Defaults to None.
            media_type (str): Media type of the stream data as specified in NOVA-DB.
            file_ext (str, optional): File extension. Defaults to None.
            dim_labels (list[dict], optional): Dimension labels. Defaults to None.
            is_valid (bool, optional): Indicates if the stream data is valid. Defaults to True.

        Raises:
            FileNotFoundError: If the data directory is not set.
        """


        # save file to disk
        dataset = dataset if not dataset is None else stream.meta_data.dataset
        session = session if not session is None else stream.meta_data.session
        role = role if not role is None else stream.meta_data.role
        name = name if not name is None else stream.meta_data.name
        file_ext = file_ext if not file_ext is None else stream.meta_data.ext
        media_type = media_type if not media_type is None else stream.meta_data.media_type

        if isinstance(stream, SSIStream):
            dim_labels = dim_labels if not dim_labels is None else stream.meta_data.dim_labels
        else:
            dim_labels = None

        if not self.data_dir:
            raise FileNotFoundError("Data directory was not set. Can't access files")

        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Specified data directory {self.data_dir} is not a directory.")

        file_name = role + "." + name + file_ext
        file_path = self.data_dir / dataset / session / file_name

        FileHandler().save(stream, file_path)

        meta_data: StreamMetaData = stream.meta_data

        # write db entry
        stream_document = {
            "fileExt": file_ext.strip('.'),
            "name": name,
            "sr": meta_data.sample_rate,
            "type": media_type,
            "dimlabels": dim_labels if dim_labels else [],
            "isValid": is_valid,
        }

        # check if stream exists
        result = self.client[dataset][STREAM_COLLECTION].find_one({"name": name})

        # update existing
        if result:
            update_query_annotation = {"$set": stream_document}
            self.client[dataset][STREAM_COLLECTION].update_one(
                {"_id": result["_id"]}, update_query_annotation
            )

        # insert new
        else:
            self.client[dataset][STREAM_COLLECTION].insert_one(stream_document)


if __name__ == "__main__":
    import os
    import random
    from time import perf_counter
    from dotenv import load_dotenv

    test_annotations = False
    test_streams = True

    load_dotenv("../../../.env")
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")
    DATA_DIR = os.getenv("NOVA_DATA_DIR", None)

    if test_annotations:
        amh = AnnotationHandler(db_host=IP, db_port=PORT, db_user=USER, db_password=PASSWORD)

        # load
        fs = "Loading {} took {}ms"
        t_start = perf_counter()
        discrete_anno = amh.load(
            dataset="test",
            scheme="diarization",
            annotator="schildom",
            session="04_Oesterreich_test",
            role="testrole2",
            header_only=True
        )
        t_stop = perf_counter()
        print(fs.format("Discrete annotation", int((t_stop - t_start) * 1000)))

        t_start = perf_counter()
        continuous_anno = amh.load(
            dataset="test",
            scheme="arousal",
            annotator="emow2v",
            session="01_AffWild2_video1",
            role="testrole",
        )
        t_stop = perf_counter()
        print(fs.format("Continuous annotation", int((t_stop - t_start) * 1000)))

        t_start = perf_counter()
        free_anno = amh.load(
            dataset="test",
            scheme="transcript",
            annotator="whisperx",
            session="04_Oesterreich_test",
            role="testrole",
        )
        t_stop = perf_counter()
        print(fs.format("Free annotation", int((t_stop - t_start) * 1000)))

        # save
        fs = "Saving {} took {}ms"
        t_start = perf_counter()
        amh.save(
            dataset="test",
            annotation=discrete_anno,
            session="04_Oesterreich_test",
            annotator="testuser",
            role="testrole",
            overwrite=True,
        )
        t_stop = perf_counter()
        print(fs.format("Discrete annotation", int((t_stop - t_start) * 1000)))

        t_start = perf_counter()
        # amh.save(continuous_anno, annotator="testuser", overwrite=True)
        t_stop = perf_counter()
        print(fs.format("Continuous annotation", int((t_stop - t_start) * 1000)))

        t_start = perf_counter()
        # amh.save(free_anno, annotator="testuser", overwrite=True)
        t_stop = perf_counter()
        print(fs.format("Free annotation", int((t_stop - t_start) * 1000)))

    if test_streams:

        smh = StreamHandler(
            db_host=IP, db_port=PORT, db_user=USER, db_password=PASSWORD, data_dir=Path(DATA_DIR)
        )

        # Stream
        fs = "Loading {} took {}ms"
        t_start = perf_counter()
        feature_stream = smh.load(
            dataset="test",
            session="04_Oesterreich_test",
            role="testrole",
            name="arousal.synchrony[testrole]",
            header_only=True
        )
        t_stop = perf_counter()
        print(fs.format("Video", int((t_stop - t_start) * 1000)))

        suffix = "_testing"
        feature_stream.sample_rate = random.uniform(0, 16000)
        smh.save(
            stream=feature_stream,
            dataset="test",
            session="04_Oesterreich_test",
            role="testrole",
            name="arousal.synchrony[testrole]" + suffix,
            media_type="video",
            dim_labels=[{"id": 1, "name": "hallo"}, {"id": 2, "name": "nope"}],
        )

        # Audio
        t_start = perf_counter()
        audio_stream = smh.load(
            dataset="test", session="01_AffWild2_video1", role="testrole", name="audio"
        )
        t_stop = perf_counter()
        print(fs.format("Audio", int((t_stop - t_start) * 1000)))

        # Video
        t_start = perf_counter()
        video_stream = smh.load(
            dataset="test", session="01_AffWild2_video1", role="testrole", name="video"
        )
        t_stop = perf_counter()
        print(fs.format("Video", int((t_stop - t_start) * 1000)))

        breakpoint()
