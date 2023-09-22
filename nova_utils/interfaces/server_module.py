from abc import ABC, abstractmethod

import numpy as np

from nova_utils.data.annotation import Annotation
from nova_utils.utils.ssi_xml_utils import ModelIO
from nova_utils.utils.string_utils import parse_nova_option_string


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
    def __init__(self, model_io: list[ModelIO], opts: dict):
        self.model = None
        self.data = None
        self.output = None
        self.options = opts
        self.model_io = model_io

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

    def process_data(self, ds_iter) -> dict:
        """Returning a dictionary that contains the original keys from the dataset iterator and a list of processed samples as value. Can be overwritten to customize the processing"""
        self.ds_iter = ds_iter

        # Get all data streams of type "input" that match an id from the modules trainer file
        processed = {
            k: []
            for k in [
                d.get("id")
                for d in ds_iter.data
                if d.get("type") == "input" and d.get("id") in [mio.io_id for mio in self.model_io]
            ]
        }

        for sample in ds_iter:
            for id, output_list in processed.items():
                data_for_id = {id: sample[id]}
                out = self.preprocess_sample(data_for_id)
                out = self.process_sample(out)
                out = self.postprocess_sample(out)
                output_list.append(out)

        return processed


class Trainer(Processor):
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
    def to_stream(self, data: object) -> dict:
        """Converts the return value from process_data() to data stream chunk that can be processed by nova-server.

        Args:
            data (object): The data as returned by the process_data function of the Processor class


        Returns:
            dict: A dictionary mapping a stream identifier (usually composed using the role, signal, extracted feature name and sliding window parameters) to a tuple containing a chunk of the data as well as additional information.
        Each tuple has the form ( type (nova_types.DataTypes), sample_rate (double), data_chunk (numpy.ndarray) ). The shape of the data chunk should in the form of (n_frames, n_features)
        An arbitrary number of streams maybe returned.

        Example:
            ::

                {
                    'speaker_1.audio.mfcc[10ms,10ms,10ms]' : ( nova_utils.db_utils.nova_types.DataTypes.AUDIO, 100, [[0.0, 0.0, ... 0.0], [0.0, 0.0, ... 0.0] ... [0.0, 0.0, ... 0.0]] ),
                    'speaker_2.audio.mfcc[10ms,10ms,10ms]' : ( nova_utils.db_utils.nova_types.DataTypes.AUDIO, 100, [[0.0, 0.0, ... 0.0], [0.0, 0.0, ... 0.0] ... [0.0, 0.0, ... 0.0]] )
                }
        """
        raise NotImplemented
