import numpy as np
import warnings
from datetime import datetime
from pymongo import MongoClient
from pymongo.results import InsertOneResult, UpdateResult
from nova_utils.data.data_handler.file_handler import FileHandler, FileMetaData
from bson.objectid import ObjectId
from pathlib import Path
from nova_utils.data.data_handler.ihandler import IHandler
from nova_utils.data.annotation import (
    LabelType,
    SchemeType,
    Annotation,
    DiscreteAnnotation,
    DiscreteAnnotationScheme,
    ContinuousAnnotation,
    ContinuousAnnotationScheme,
    FreeAnnotation,
    FreeAnnotationScheme,
)
from nova_utils.data.stream import Stream, SSIStream, Video, Audio


ANNOTATOR_COLLECTION = "Annotators"
SCHEME_COLLECTION = "Schemes"
STREAM_COLLECTION = "Streams"
ROLE_COLLECTION = "Roles"
ANNOTATION_COLLECTION = "Annotations"
SESSION_COLLECTION = "Sessions"
ANNOTATION_DATA_COLLECTION = "AnnotationData"

# METADATA
class MongoMetaData():
    def __init__(self, ip: str = None, port: int = None, user: str = None, dataset: str = None):
        self.ip = ip
        self.port = port
        self.user = user
        self.dataset = dataset


class MongoAnnotationMetaData(MongoMetaData):
    def __init__(
        self,
        is_locked: bool = None,
        is_finished: bool = None,
        last_update: bool = None,
        annotation_document_id: ObjectId = None,
        data_document_id: ObjectId = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_locked = is_locked
        self.is_finished = is_finished
        self.last_update = last_update
        self.annotation_document_id = annotation_document_id
        self.data_document_id = data_document_id


class MongoStreamMetaData(MongoMetaData):
    def __init__(
        self,
        name: str = None,
        dim_labels: dict = None,
        file_ext: str = None,
        is_valid: bool = None,
        stream_document_id: ObjectId = None,
        sr: float = None,
        type: str = None,
        file_handler_meta_data: FileMetaData = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.dim_labels = dim_labels
        self.file_ext = file_ext
        self.is_valid = is_valid
        self.stream_document_id = stream_document_id
        self.db_sample_rate = sr
        self.type = type
        self.file_handler_meta_data = file_handler_meta_data


# DATA
class _IMongoHandler:
    def __init__(
        self,
        ip: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        data_dir: Path = None,
    ):
        self._client = None
        self._ip = None
        self._port = None
        self._user = None
        self.data_dir = data_dir
        if ip and port and user and password:
            self.connect(ip, port, user, password)

    def connect(
        self, ip: str = None, port: int = None, user: str = None, password: str = None
    ):
        self._client = MongoClient(host=ip, port=port, username=user, password=password)
        self._ip = ip
        self._port = port
        self._user = user

    @property
    def client(self):
        if self._client is None:
            raise ValueError(
                "Connection to mongo DB is not established. Call connect() first."
            )
        return self._client


class AnnotationHandler(IHandler, _IMongoHandler):
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

    def _update_annotation(
        self,
        dataset: str,
        annotation_id: ObjectId,
        annotation_data_id: ObjectId,
        annotation_data: list,
        is_finished: bool,
        is_locked: bool,
    ) -> UpdateResult:

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
        self, dataset: str, scheme: str, session: str, annotator: str, role: str
    ) -> Annotation:

        # load annotation from mongo db
        anno_doc = self._load_annotation(dataset, session, annotator, role, scheme)

        if not anno_doc:
            raise FileNotFoundError(
                f"Annotation not found \ndataset: {dataset} \nsession: {session} \nannotator: {annotator} \nrole: {role} \nscheme: {scheme}"
            )

        (anno_data_doc,) = anno_doc["data"]

        # build annotation object
        (scheme_doc,) = anno_doc["scheme"]
        scheme_type = scheme_doc["type"]

        # discrete scheme
        if scheme_type == SchemeType.DISCRETE.name:
            scheme_classes = {l["id"]: l["name"] for l in scheme_doc["labels"]}

            anno_data = np.array(
                [
                    (x["from"], x["to"], x["id"], x["conf"])
                    for x in anno_data_doc["labels"]
                ],
                dtype=LabelType.DISCRETE.value,
            )
            anno_scheme = DiscreteAnnotationScheme(name=scheme, classes=scheme_classes)
            annotation = DiscreteAnnotation(
                #role=role,
                #annotator=annotator,
                #annotation_scheme=anno_scheme,
                #session=session,
                #dataset=dataset,
                data=anno_data,
                scheme=anno_scheme
            )

        # continuous scheme
        elif scheme_type == SchemeType.CONTINUOUS.name:
            sr = scheme_doc["sr"]
            min_val = scheme_doc["min"]
            max_val = scheme_doc["max"]
            anno_data = np.array(
                [(x["score"], x["conf"]) for x in anno_data_doc["labels"]],
                dtype=LabelType.CONTINUOUS.value,
            )
            anno_scheme = ContinuousAnnotationScheme(
                name=scheme, sample_rate=sr, min_val=min_val, max_val=max_val
            )
            annotation = ContinuousAnnotation(
                #role=role,
                #annotator=annotator,
                #annotation_scheme=anno_scheme,
                #session=session,
                #dataset=dataset,
                data=anno_data,
                scheme=anno_scheme
            )

            # free scheme
        elif scheme_type == SchemeType.FREE.name:
            anno_data = np.array(
                [
                    (x["from"], x["to"], x["name"], x["conf"])
                    for x in anno_data_doc["labels"]
                ],
                dtype=LabelType.FREE.value,
            )
            anno_scheme = FreeAnnotationScheme(name=scheme)
            annotation = FreeAnnotation(
                #role=role,
                #annotator=annotator,
                #annotation_scheme=anno_scheme,
                #session=session,
                #dataset=dataset,
                data=anno_data,
                scheme=anno_scheme
            )
        else:
            raise TypeError(f"Unknown scheme type {scheme_type}")

        # handler meta data
        # handler_meta_data = MongoAnnotationMetaData(
        #     ip=self._ip,
        #     port=self._port,
        #     user=self._user,
        #     is_locked=anno_doc.get("isLocked"),
        #     is_finished=anno_doc.get("isFinished"),
        #     annotation_document_id=anno_doc.get("_id"),
        #     data_document_id=anno_doc.get("data_id"),
        #     last_update=anno_doc.get("date"),
        # )
        # annotation.info.handler = handler_meta_data

        # setting meta data
        return annotation

    def save(
        self,
        annotation: Annotation,
        dataset: str,
        session: str,
        annotator: str,
        role: str,
        is_finished: bool = False,
        is_locked: bool = False,
        overwrite: bool = False,
    ):

        # overwrite default values
        dataset = dataset
        session = session
        annotator = annotator
        role = role
        scheme = annotation.annotation_scheme.name

        # TODO check for none values
        anno_data = [
            dict(zip(annotation.annotation_scheme.label_dtype.names, ad.item()))
            for ad in annotation.data
        ]

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
                warnings.warn(
                    f"Can't overwrite locked annotation \ndataset: {dataset} \nsession: {session} \nannotator: {annotator} \nrole: {role} \nscheme: {scheme}"
                )
                return None
            elif not overwrite:
                warnings.warn(
                    f"Can't overwrite annotation \ndataset: {dataset} \nsession: {session} \nannotator: {annotator} \nrole: {role} \nscheme: {scheme}. Because overwrite is disabled."
                )
                return None
            else:
                warnings.warn(
                    f"Overwriting existing annotation \ndataset: {dataset} \nsession: {session} \nannotator: {annotator} \nrole: {role} \nscheme: {scheme}"
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


class StreamHandler(IHandler, _IMongoHandler):
    def _load_stream(
        self,
        dataset: str,
        stream_name: str,
    ) -> dict:

        result = self.client[dataset][STREAM_COLLECTION].find_one({"name": stream_name})
        if not result:
            return {}
        return result

    def load(self, dataset: str, session: str, role: str, name: str) -> Stream:
        result = self._load_stream(dataset=dataset, stream_name=name)
        if not result:
            raise ValueError(f"No stream {name} found for dataset {dataset}")
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
        data = FileHandler().load(file_path)
        assert isinstance(data, Stream)

        # meta data
        # handler_meta_data = MongoStreamMetaData(
        #     ip=self._ip,
        #     port=self._port,
        #     user=self._user,
        #     name=result.get("name"),
        #     dim_labels=result.get("dimLabels"),
        #     file_ext=result.get("fileExt"),
        #     is_valid=result.get("isValid"),
        #     stream_document_id=result.get("_id"),
        #     sr=result.get("sr"),
        #     type=result.get("type"),
        #     file_handler_meta_data=data.info.handler,
        # )
        # data.info.handler = handler_meta_data

        return data

    def save(self,
             stream: Stream,
             dataset: str,
             session: str,
             role: str,
             name: str,
             data_type: str,
             file_ext: str = None,
             dim_labels: [] = None,
             is_valid: bool = True,

             ):

        if not self.data_dir:
            raise FileNotFoundError("Data directory was not set. Can't access files")

        # write file
        if file_ext is None:
            if isinstance(stream, SSIStream):
                file_ext = 'stream'
            elif isinstance(stream, Audio):
                file_ext = 'wav'
            elif isinstance(stream, Video):
                file_ext = 'mp4'

        file_name = (role + "." + name + "." + file_ext)
        file_path = Path(
            self.data_dir
            / dataset
            / session
            / file_name
        )

        FileHandler().save(stream, file_path)

        # write db entry
        stream_document = {
            "fileExt" : file_ext,
            "name" : name,
            "sr" : stream.sample_rate,
            "type" : data_type,
            "dimlabels" : dim_labels if dim_labels else [],
            "isValid" : is_valid
        }

        # check if stream exists
        result = self.client[dataset][STREAM_COLLECTION].find_one({"name": name})

        # update existing
        if result:
            update_query_annotation = {
                "$set": stream_document
            }
            self.client[dataset][STREAM_COLLECTION].update_one(
                {"_id": result["_id"]}, update_query_annotation
            )

        # insert new
        else:
            self.client[dataset][STREAM_COLLECTION].insert_one(
                stream_document
            )




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
        amh = AnnotationHandler(ip=IP, port=PORT, user=USER, password=PASSWORD)

        # load
        fs = "Loading {} took {}ms"
        t_start = perf_counter()
        discrete_anno = amh.load(
            dataset="test",
            scheme="diarization",
            annotator="schildom",
            session="04_Oesterreich_test",
            role="testrole2",
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
        #amh.save(discrete_anno, annotator="testuser", overwrite=True)
        t_stop = perf_counter()
        print(fs.format("Discrete annotation", int((t_stop - t_start) * 1000)))

        t_start = perf_counter()
        #amh.save(continuous_anno, annotator="testuser", overwrite=True)
        t_stop = perf_counter()
        print(fs.format("Continuous annotation", int((t_stop - t_start) * 1000)))

        t_start = perf_counter()
        #amh.save(free_anno, annotator="testuser", overwrite=True)
        t_stop = perf_counter()
        print(fs.format("Free annotation", int((t_stop - t_start) * 1000)))

    if test_streams:

        smh = StreamHandler(
            ip=IP, port=PORT, user=USER, password=PASSWORD, data_dir=Path(DATA_DIR)
        )

        # Loading
        fs = "Loading {} took {}ms"
        t_start = perf_counter()
        feature_stream = smh.load(
            dataset="test",
            session="04_Oesterreich_test",
            role="testrole",
            name="arousal.synchrony[testrole]",
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
            data_type='video',
            dim_labels=[{'1' : 'hallo'}, {'2' : 'nope'}]
        )


        t_start = perf_counter()
        audio_stream = smh.load(
            dataset="test", session="01_AffWild2_video1", role="testrole", name="audio"
        )
        t_stop = perf_counter()
        print(fs.format("Audio", int((t_stop - t_start) * 1000)))

        t_start = perf_counter()
        video_stream = smh.load(
            dataset="test", session="01_AffWild2_video1", role="testrole", name="video"
        )
        t_stop = perf_counter()
        print(fs.format("Video", int((t_stop - t_start) * 1000)))


        breakpoint()
