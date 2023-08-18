from abc import ABC, abstractmethod

from nova_utils.ssi_utils.ssi_anno_utils import Anno


REQUIREMENTS = []

class Processor(ABC):
    """
    Base class of a data processor. This interface builds the foundation for all data processing classes.
    """

    # List of dependencies that need to be installed when the script is loaded
    DEPENDENCIES = []

    # Flag to indicate whether the processed input belongs to one role or to multiple roles
    SINGLE_ROLE_INPUT = True

    def __init__(self, logger, request_form=None):
        self.logger = logger
        self.request_form = request_form
        self.model = None
        self.data = None
        self.output = None
        self.options = {}

        if self.request_form is not None:
            # Set Options
            logger.info("Setting options...")
            if request_form.get("optStr"):
                for k, v in dict(
                        option.split("=") for option in request_form["optStr"].split(";")
                ).items():
                    if v in ("True", "False"):
                        self.options[k] = True if v == "True" else False
                    elif v == "None":
                        self.options[k] = None
                    else:
                        self.options[k] = v
                    logger.info(k + "=" + v)
            logger.info("...done.")

    @abstractmethod
    def preprocess_sample(self, sample: dict ):
        """Preprocess data to convert between nova-server dataset iterator item to the raw model input as required in process_sample.

        Args:
            sample (dict):
        """
        return sample

    @abstractmethod
    def process_sample(self, sample):
        """Applying processing steps (e.g. feature extraction, data prediction etc... ) to the provided data."""
        return sample

    @abstractmethod
    def postprocess_sample(self, sample):
        """Apply any optional postprocessing to the data (e.g. scaling, mapping etc...)"""
        return sample

    def process_data(self, ds_iter) -> dict:
        """Returning a dictionary that contains the original keys from the dataset iterator and processed samples as value. Can be overwritten to customize the processing"""
        self.ds_iter = ds_iter
        processed = {k: [] for k in ds_iter.data_schemes.keys()}

        for sample in ds_iter:
            for k in ds_iter.data_schemes.keys():
                out = self.preprocess_sample(sample[k])
                out = self.process_sample(out)
                out = self.postprocess_sample(out)
                processed[k].append(out)

        return processed


class Trainer(Processor):
    """
    Base class of a Trainer. Implement this interface in your own class to build a model that is trainable from within nova
    """

    """Includes all the necessary files to run this script"""

    def __init__(self, logger, request_form=None):
        super().__init__(logger, request_form)

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

    def __init__(self, logger, request_form=None):
        super().__init__(logger, request_form)

    @abstractmethod
    def to_anno(self, data) -> list[Anno]:
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

    def __init__(self, logger, request_form=None):
        super().__init__(logger, request_form)

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
