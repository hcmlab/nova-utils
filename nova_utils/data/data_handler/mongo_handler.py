import numpy as np
import warnings
from datetime import datetime
from pymongo import MongoClient
from pymongo.results import InsertOneResult, UpdateResult
from typing import List
from bson.objectid import ObjectId
from nova_utils.data.data_handler.ihandler import IHandler
from nova_utils.data.annotation import (
    LabelType,
    SchemeType,
    IAnnotation,
    DiscreteAnnotation,
    DiscreteAnnotationScheme,
    ContinuousAnnotation,
    ContinuousAnnotationScheme,
    FreeAnnotation,
    FreeAnnotationScheme,
)

ANNOTATOR_COLLECTION = "Annotators"
SCHEME_COLLECTION = "Schemes"
STREAM_COLLECTION = "Streams"
ROLE_COLLECTION = "Roles"
ANNOTATION_COLLECTION = "Annotations"
SESSION_COLLECTION = "Sessions"
ANNOTATION_DATA_COLLECTION = "AnnotationData"


class MongoHandler(IHandler):
    def __init__(
        self, ip: str = None, port: int = None, user: str = None, password: str = None
    ):
        self._client = None
        if ip and port and user and password:
            self._client = MongoClient(
                host=ip, port=port, username=user, password=password
            )

    def connect(
        self, ip: str = None, port: int = None, user: str = None, password: str = None
    ):
        self._client = MongoClient(host=ip, port=port, username=user, password=password)

    @property
    def client(self):
        if self._client is None:
            raise ValueError(
                "Connection to mongo DB is not established. Call connect() first."
            )
        return self._client


class AnnotationMongoHandler(MongoHandler):
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
    ) -> IAnnotation:

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
                role=role,
                annotator=annotator,
                annotation_scheme=anno_scheme,
                session=session,
                dataset=dataset,
                data=anno_data,
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
                role=role,
                annotator=annotator,
                annotation_scheme=anno_scheme,
                session=session,
                dataset=dataset,
                data=anno_data,
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
                role=role,
                annotator=annotator,
                annotation_scheme=anno_scheme,
                session=session,
                dataset=dataset,
                data=anno_data,
            )
        else:
            raise TypeError(f"Unknown scheme type {scheme_type}")

        return annotation

    def save(
        self,
        annotation: IAnnotation,
        dataset: str = None,
        session: str = None,
        annotator: str = None,
        role: str = None,
        is_finished: bool = False,
        is_locked: bool = False,
        overwrite: bool = False,
    ):

        # overwrite default values
        dataset = dataset if dataset else annotation.meta_info.dataset
        session = session if session else annotation.meta_info.session
        annotator = annotator if annotator else annotation.meta_info.annotator
        role = role if role else annotation.meta_info.role
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

        # TODO success error handling


class StreamDataHandler(MongoHandler):
    """
    Class for handling download of data streams from Mongo db.
    """

    def load(self, dataset: str, name: str):
        """
        Load data streams from Mongo db based on the provided parameters.

        Parameters:
            dataset (str): The dataset name.
            name (str): The name of the data stream.

        Returns:
            List[DataStream]: A list of data stream objects.
        """
        # Retrieve data streams from Mongo db using provided parameters
        # Placeholder: Replace with the actual Mongo db query based on your implementation
        data_streams = self.db_connection.get_data_streams(dataset, name)

        return data_streams

    def save(self, data_streams):
        """
        Save data streams to Mongo db.

        Parameters:
            data_streams (List[DataStream]): A list of data stream objects to save.

        Returns:
            None
        """
        # Save data streams to Mongo db
        # Placeholder: Replace with the actual Mongo db save method based on your implementation
        self.db_connection.save_data_streams(data_streams)


if __name__ == "__main__":
    import os
    from time import perf_counter
    from dotenv import load_dotenv

    load_dotenv("../../../.env")
    IP = os.getenv("NOVA_IP", "")
    PORT = int(os.getenv("NOVA_PORT", 0))
    USER = os.getenv("NOVA_USER", "")
    PASSWORD = os.getenv("NOVA_PASSWORD", "")

    amh = AnnotationMongoHandler(ip=IP, port=PORT, user=USER, password=PASSWORD)

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
    amh.save(discrete_anno, annotator="testuser", overwrite=True)
    t_stop = perf_counter()
    print(fs.format("Discrete annotation", int((t_stop - t_start) * 1000)))

    fs = "Saving {} took {}ms"
    t_start = perf_counter()
    amh.save(continuous_anno, annotator="testuser", overwrite=True)
    t_stop = perf_counter()
    print(fs.format("Continuous annotation", int((t_stop - t_start) * 1000)))

    fs = "Saving {} took {}ms"
    t_start = perf_counter()
    amh.save(free_anno, annotator="testuser", overwrite=True)
    t_stop = perf_counter()
    print(fs.format("Free annotation", int((t_stop - t_start) * 1000)))