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

    def _get_annotation(
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
        self, dataset: str, anno_id: ObjectId, anno_data_id: ObjectId, anno: IAnnotation
    ) -> tuple[UpdateResult, UpdateResult]:

        # Todo consider mechanism to also update isLocked and isFinished attributes
        update_query_annotation = {"$set": {"date": datetime.now()}}
        update_query_annotation_data = {
            "$set": {
                "labels": [
                    dict(zip(anno.annotation_scheme.label_dtype.names, ad.item()))
                    for ad in anno.data
                ]
            }
        }
        result_anno = self.client[dataset][ANNOTATION_COLLECTION].update_one(
            {"_id": anno_id}, update_query_annotation
        )
        result_anno_data = self.client[dataset][ANNOTATION_DATA_COLLECTION].update_one(
            {"_id": anno_data_id}, update_query_annotation_data
        )

        return (result_anno, result_anno_data)

    def _insert_annotation_data(
        self, dataset: str, annotation_data_doc: dict
    ) -> InsertOneResult:
        result = self.client[dataset][ANNOTATION_DATA_COLLECTION].insert_one(
            annotation_data_doc
        )
        return result

    def load(
        self, dataset: str, scheme: str, session: str, annotator: str, role: str
    ) -> IAnnotation:

        # load annotation from mongo db
        anno_doc = self._get_annotation(dataset, session, annotator, role, scheme)

        if not anno_doc:
            raise FileNotFoundError(
                f"Can't overwrite locked annotation \ndataset: {dataset} \nsession: {session} \nannotator: {annotator} \nrole: {role} \nscheme: {scheme}"
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
        """
        Save annotation data to Mongo db.

        Parameters:
            annotation_data (List[AnnotationData]): A list of annotation data objects to save.

        Returns:
            None
        """
        # overwrite default values
        dataset = dataset if dataset else annotation.meta_info.dataset
        session = session if session else annotation.meta_info.session
        annotator = annotator if annotator else annotation.meta_info.annotator
        role = role if role else annotation.meta_info.role
        scheme = annotation.annotation_scheme.name

        # TODO check for none values

        # get documents
        anno_doc = self._get_annotation(
            dataset,
            session,
            annotator,
            role,
            scheme,
            project={"_id": 1, "isLocked": 1, "data_id": 1},
        )

        # if anno already exists
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
                    dataset,
                    anno_id=anno_doc["_id"],
                    anno_data_id=anno_doc["data_id"],
                    anno=annotation,
                )
                return success

        # TODO: Do it!

        # new anno
        else:
            success = self._insert_annotation_data(dataset, anno_data_doc)
            if not success.acknowledged:
                warnings.warn(
                    f"Unexpected error uploading annotation \ndataset: {dataset} \nsession: {session} \nannotator: {annotator} \nrole: {role} \nscheme: {scheme}"
                )
                return None
            else:
                anno_data_doc_id = success.inserted_id

            anno_doc = {
                "data_id": anno_data_doc_id,
                "annotator_id": annotator_doc[_id],
                "role_id": mongo_role[0]["_id"],
                "scheme_id": mongo_scheme[0]["_id"],
                "session_id": mongo_session[0]["_id"],
                "isFinished": is_finished,
                "isLocked": is_locked,
                "date": datetime.today().replace(microsecond=0),
            }

        success = self.update_doc_by_prop(
            doc=mongo_label_doc,
            database=database,
            collection=self.ANNOTATION_DATA_COLLECTION,
        )
        if not success.acknowledged:
            warnings.warn(
                f"Unknown error update database entries for Annotation data {mongo_data_id}"
            )
            return ""
        else:
            data_id = mongo_data_id

        # anno_data_doc = self._get_annotation_data(dataset, anno_doc["data_id"])


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
        role="testrole",
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
    amh.save(discrete_anno, overwrite=True)
