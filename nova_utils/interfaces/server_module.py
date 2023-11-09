import typing
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Type
from nova_utils.data.annotation import Annotation
from nova_utils.data.provider.nova_dataset_iterator import NovaDatasetIterator
from nova_utils.data.provider.data_manager import DatasetManager, SessionManager
from nova_utils.data.stream import Stream
from nova_utils.utils.log_utils import log
from nova_utils.utils.ssi_xml_utils import ModelIO
from nova_utils.utils.ssi_xml_utils import Trainer as SSITrainer


class Processor(ABC):
    """
    Base class of a data processor. This interface builds the foundation for all data processing classes.
    """

    # List of dependencies that need to be installed when the script is loaded
    DEPENDENCIES = []

    # Flag to indicate whether the processed input belongs to one role or to multiple roles
    SINGLE_ROLE_INPUT = True

    # Prefix for provided datastreams that are missing the id-tag
    UNKNOWN_ID = '<unk>_'

    # TODO read trainer or chain file for default options
    def __init__(self, model_io: list[ModelIO], opts: dict, trainer: SSITrainer = None):
        self.model = None
        self.data = None
        self.output = None
        self.options = opts
        self.model_io = model_io
        self.trainer = trainer

    def get_session_manager(self, dataset_manager: DatasetManager, session_name: str = None, ignore_ambiguity: bool = False) -> SessionManager:
        """Returns a session manager that matches the provides session name. If no session name is provided the first

        Args:
            dataset_manager (DatasetManager): The dataset manager that holds all sessions
            session_name (str, Optional): The name of a specific session to retrieve. Defaults to None.
            ignore_ambiguity (bool, Optional): If set to False and no session name is specified and more than one session exists the function throws an error. Else the first session in the dictionary will be returned.

        Raises:
            ValueError: If session to return cannot be identified
        """
        current_session_name = session_name
        if current_session_name is None:
            if len(dataset_manager.session_names) == 1 or ignore_ambiguity:
                current_session_name = dataset_manager.session_names[0]
            else:
                raise ValueError('')

        return dataset_manager.sessions[current_session_name]['manager']

    def preprocess_sample(self, sample: dict):
        """Preprocess data to convert between nova-server dataset iterator item to the raw model input as required in process_sample.

        Args:
            sample :
        """
        return list(sample.values())[0]

    def process_sample(self, sample):
        """Applying processing steps (e.g. feature extraction, data prediction etc... ) to the provided data."""
        return sample

    def postprocess_sample(self, sample):
        """Apply any optional postprocessing to the data (e.g. scaling, mapping etc...)"""
        return sample

    def process_data(self, ds_manager: DatasetManager) -> dict:
        """Returning a dictionary that contains the original keys from the dataset iterator and a list of processed
        samples as value. Can be overwritten to customize the processing"""
        ds_manager: NovaDatasetIterator

        # Start the stopwatch / counter
        pc_start = perf_counter()
        output_list = []
        for i, sample in enumerate(ds_manager):
            if i % 100 == 0:
                log(f'Processing sample {i}. {i / (perf_counter() - pc_start)} samples / s. Processed {i * ds_manager.stride / 1000} Seconds of data.')
            # for id, output_list in processed.items():
            #     data_for_id = {id: sample[id]}
            out = self.preprocess_sample(sample)
            out = self.process_sample(out)
            out = self.postprocess_sample(out)
            output_list.append(out)


        # TODO convert output to dict
        return output_list

    def to_output(self, data):
        if isinstance(self, Predictor):
            return self.to_anno(data)
        elif isinstance(self, Extractor):
            return self.to_stream(data)

class Trainer(ABC):
    """
    Base class of a Trainer. Implement this interface in your own class to build a model that is trainable from within nova
    """

    """Includes all the necessary files to run this script"""

    @abstractmethod
    def train(self):
        """Trains a model with the given data."""
        raise NotImplemented

    @abstractmethod
    def save(self, path) -> str:
        """Stores the weights of the given model at the given path. Returns the path of the weights."""
        raise NotImplemented

    @abstractmethod
    def load(self, path):
        """Loads a model with the given path. Returns this model."""
        raise NotImplemented

class Explainer(ABC):
    @abstractmethod
    def get_explainable_model(self) -> typing.Any:
        raise NotImplemented
    @abstractmethod
    def get_predict_function(self) -> typing.Callable:
        raise NotImplemented

class Predictor(Processor):
    """
    Base class of a data predictor. Implement this interface if you want to write annotations to a database
    """

    @abstractmethod
    def to_anno(self, data) -> list[Annotation]:
        """Converts the output of process_data to the correct annotation format to upload them to the database.
        !THE OUTPUT FORMAT OF THIS FUNCTION IS NOT YET FULLY DEFINED AND WILL CHANGE IN FUTURE RELEASES!

        Args:
            data (object): Data output of process_data function

        Returns:
            list: A list of annotation objects
        """
        raise NotImplemented


class Extractor(Processor):
    """
    Base class of a feature extractor. Implement this interface in your own class to build a feature extractor.
    """

    @property
    @abstractmethod
    def chainable(self):
        """Whether this extraction module can be followed by other extractors. If set to True 'to_ds_iterable()' must be implemented"""
        return False

    @abstractmethod
    def to_stream(self, data: object) -> list[Stream]:
        """Converts the return value from process_data() to data stream chunk that can be processed by nova-server.

        Args:
            data (object): The data as returned by the process_data function of the Processor class


        Returns:
            list: A list of stream objects.

        """
        raise NotImplemented
