from nova_utils.data.data_handler.ihandler import IHandler
import numpy as np
from pymongo import MongoClient
from typing import List
from bson.objectid import ObjectId
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

    def _find_one_by_name(self, name: str, collection: str, dataset: str):
        query_filter = {"name": name}
        result = self.client[dataset][collection].find_one(query_filter)
        if result:
            return result
        else:
            raise FileNotFoundError(
                f'Could not find any entry for name "{name}" in collection {collection} in database {dataset}'
            )

    def _get_session(self, dataset: str, session: str) -> dict:
        return self._find_one_by_name(session, SESSION_COLLECTION, dataset)

    def _get_scheme(self, dataset: str, scheme: str) -> dict:
        return self._find_one_by_name(scheme, SCHEME_COLLECTION, dataset)

    def _get_role(self, dataset: str, role: str) -> dict:
        return self._find_one_by_name(role, ROLE_COLLECTION, dataset)

    def _get_annotator(self, dataset: str, annotator: str) -> dict:
        return self._find_one_by_name(annotator, ANNOTATOR_COLLECTION, dataset)

    def _get_anno_data(self, dataset: str, anno_id: ObjectId) -> dict:
        query_filter = {"_id": anno_id}
        result = self.client[dataset][ANNOTATION_DATA_COLLECTION].find_one(query_filter)
        if result:
            return result
        else:
            raise FileNotFoundError(
                f'Could not find any entry for _id "{anno_id}" in collection {ANNOTATION_DATA_COLLECTION} in database {dataset}'
            )

    def load(
        self, dataset: str, scheme: str, session: str, annotator: str, role: str
    ) -> IAnnotation:
        """
        Load annotation data from Mongo db based on the provided parameters.

        Parameters:
            dataset (str): The dataset name.
            scheme (str): The scheme name.
            session (str): The session name.
            annotator (str): The annotator name.
            role (str): The roles for which annotations are requested.

        Returns:
            List[AnnotationData]: A list of annotation data objects.
        """

        # load annotation from mongo db
        session_object = self._get_session(dataset, session)
        scheme_object = self._get_scheme(dataset, scheme)
        annotator_object = self._get_annotator(dataset, annotator)
        role_object = self._get_role(dataset, role)

        query_filter = {
            "$and": [
                {"session_id": session_object["_id"]},
                {"scheme_id": scheme_object["_id"]},
                {"annotator_id": annotator_object["_id"]},
                {"role_id": role_object["_id"]},
            ]
        }
        anno_doc = self.client[dataset][ANNOTATION_COLLECTION].find_one(query_filter)
        if not anno_doc:
            raise FileNotFoundError(
                f'Could not find any entry for filter "{query_filter}" in collection {ANNOTATION_COLLECTION} in database {dataset}'
            )

        anno_data_doc = self._get_anno_data(dataset, anno_doc["data_id"])

        # build annotation object
        scheme_type = scheme_object["type"]

        # discrete scheme
        if scheme_type == SchemeType.DISCRETE.name:
            scheme_classes = {}
            for item in scheme_object["labels"]:
                scheme_classes[item.get("id")] = item.get("name")

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
            sr = scheme_object["sr"]
            min_val = scheme_object["min"]
            max_val = scheme_object["max"]
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
            raise TypeError(f"Unknown scheme type {type}")

        return annotation

    def save(self, annotation: str, session: str, dataset: str):
        """
        Save annotation data to Mongo db.

        Parameters:
            annotation_data (List[AnnotationData]): A list of annotation data objects to save.

        Returns:
            None
        """
        # Save annotation data to Mongo db
        # Placeholder: Replace with the actual Mongo db save method based on your implementation
        self.db_connection.save_annotation_data(annotation_data)


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
    ip = "137.250.171.233"
    port = 37317
    user = "schildom"
    password = "HsE8aFk8G.hGYzCe4dE"

    amh = AnnotationMongoHandler(ip=ip, port=port, user=user, password=password)

    discrete_anno = amh.load(
        dataset="test",
        scheme="diarization",
        annotator="schildom",
        session="04_Oesterreich_test",
        role="testrole",
    )
    continuous_anno = amh.load(
        dataset="test",
        scheme="arousal",
        annotator="emow2v",
        session="01_AffWild2_video1",
        role="testrole",
    )
    free_anno = amh.load(
        dataset="test",
        scheme="transcript",
        annotator="whisperx",
        session="04_Oesterreich_test",
        role="testrole",
    )
