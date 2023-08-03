from abc import ABC, abstractmethod


class DatasetIterable(ABC):
    """
    Abstract superclass for iterable Datasets, which can be fed into the Bridge classes for tensorflow and pytorch
    """

    def __init__(self, *args, split: str = "train", **kwargs):
        if split not in ["test", "train", "val"]:
            split = "train"
        self.split = split

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __next__(self):
        raise NotImplementedError()

    @abstractmethod
    def get_output_info(self):
        raise NotImplementedError()
