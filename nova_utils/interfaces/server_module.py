from abc import ABC, abstractmethod

REQUIREMENTS = []


class Processor(ABC):
    """
    Base class of a data processor. This interface builds the foundation for all data processing classes.
    """

    def __init__(self, ds_iter, logger, request_form=None):
        self.model = None
        self.ds_iter = ds_iter
        self.logger = logger
        self.data = None
        self.DEPENDENCIES = []
        self.OPTIONS = {}
        self.request_form = request_form
        self.output = None

    @abstractmethod
    def preprocess_sample(self, sample: dict = None):
        """Preprocess data to convert between nova-server dataset iterator item to the raw model input as required in forward_sample."""
        return sample

    @abstractmethod
    def process_sample(self, sample):
        """Applying processing steps (e.g. feature extraction, data prediction etc... ) to the provided data."""
        return sample

    @abstractmethod
    def postprocess_sample(self, sample):
        """Apply any optional postprocessing to the data (e.g. scaling, mapping etc...)"""
        return sample

    @abstractmethod
    def process_data(self):

        processed = []

        for sample in self.ds_iter:
            out = self.preprocess_sample(sample)
            out = self.process_sample(out)
            out = self.postprocess_sample(out)
            processed.append(out)

        return processed


class Trainer(Processor):
    """
    Base class of a Trainer. Implement this interface in your own class to build a model that is trainable from within nova
    """

    """Includes all the necessary files to run this script"""

    def __init__(self, ds_iter, logger, request_form=None):
        super().__init__(ds_iter, logger, request_form)

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
    def to_anno(self, data):
        """Converts the output of process_data to the correct annotation format to upload them to the database"""
        raise NotImplemented


class Extractor(Processor):
    """
    Base class of a feature extractor. Implement this interface in your own class to build a feature extractor.
    """

    @abstractmethod
    def to_stream(self, data):
        """Converts the output of process_data to one coherent stream object that can be written to disk."""
        raise NotImplemented
