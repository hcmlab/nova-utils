"""Basic Parent class for all data types and metadata
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 18.8.2023
"""
from abc import ABC, abstractmethod

import numpy as np


class MetaData:
    """
    A class representing metadata associated with data instances.

    Metadata provides additional information about the dataset,
    the role of the data, and the session it belongs to.
    While other modules might rely on metadata information it is important to keep in mind that all metadata is purely optional.
    Always test a metadata-attribute for None.
    MetaData can also be extended by arbitrary classes using the expand method.

    Attributes:
        dataset (str): Name of the dataset.
        role (str): Role of the data (e.g., training, testing).
        session (str): Session identifier.
        name (str): Unique name to identify the data

    Args:
        dataset (str, optional): Name of the dataset the data belongs to.
        role (str, optional): Role of the data (e.g., training, testing).
        session (str, optional): Session identifier for the data.
        name (str, optional): Unique name to identify the data

    """

    def __init__(self, dataset: str = None, role: str = None, session: str = None, name: str = None):
        """
        Initialize a MetaData instance with dataset, role, and session information.
        """
        self.dataset = dataset
        self.role = role
        self.session = session
        self.name = name

    def expand(self, obj_instance):
        """
        Expand the current MetaData instance with attributes from another object.

        This method allows the MetaData instance to inherit attributes from
        another object while maintaining its original class hierarchy.

        Args:
            obj_instance: Another object whose attributes will be inherited.
        """
        new_type = type("Meta", (self.__class__, obj_instance.__class__), {})
        self.__class__ = new_type
        self.__dict__.update(obj_instance.__dict__)


class Data:
    """
    A class representing generic data along with associated metadata.

    Attributes:
        _data (np.ndarray): The underlying data array.
        meta_data (MetaData): Metadata associated with the data.

    Args:
        data (np.ndarray, optional): The data to be stored.
        dataset (str, optional): Name of the dataset the data belongs to.
        role (str, optional): ID of a subject in the dataset the data belongs to.
        session (str, optional): Session identifier for the data.
    """

    def __init__(
        self,
        data: np.ndarray = None,
        dataset: str = None,
        role: str = None,
        session: str = None,
        name: str = None,
        **kwargs
    ):
        """
        Initialize a Data instance with data and metadata.
        """
        self._data = data
        self.meta_data = MetaData(dataset, role, session, name)

    @property
    def data(self) -> np.ndarray:
        """
        Get the underlying data array. Can be overwritten by subclasses to apply custom data processing.

        Returns:
            np.ndarray: The data array.
        """
        return self._data

    @data.setter
    def data(self, value):
        """
        Set the underlying data array. Can be overwritten by subclasses to apply custom data processing.
        If the data setter is handled by the subclass th data array does not need be passed on to the parent classes.

        Args:
            value (np.ndarray): The new data array.
        """
        self._data = value

class DynamicData(ABC, Data):
    """
    An abstract subclass of Data representing dynamic data.

    This class extends the Data class and introduces an abstract method
    for sampling data from within a given interval.

    Attributes:
        (No additional attributes specified in the provided code.)

    Args:
        (No additional arguments specified in the provided code.)
    """

    @abstractmethod
    def sample_from_interval(self, start: int, end: int) -> np.ndarray:
        """
        Abstract method to sample data from within the specified interval.

        Args:
           start (int): The start index of the interval.
           end (int): The end index of the interval.

        Returns:
           np.ndarray: The sampled data within the interval.
        """
        pass


if __name__ == "__main__":
    """
    Example usage demonstrating metadata expansion.

    This block of code demonstrates how the 'MetaData' class can be used to expand
    its own attributes and methods with those from other classes, creating hybrid
    instances with combined functionality.
    """

    class MetaA:
        """
        Example class with a single attribute 'a_'.
        """

        def __init__(self):
            self.a_ = 0

    class MetaB:
        """
        Example class with an optional 'b_' attribute.

        Args:
            b (optional): An integer value for the 'b_' attribute.
        """

        def __init__(self, b=1):
            self.b_ = b

    # Create a 'MetaData' instance
    meta = MetaData()

    # Expand 'MetaData' with 'MetaB' instance
    meta.expand(MetaB(5))

    # Expand 'MetaData' with 'MetaA' instance
    meta.expand(MetaA())

    # Expand 'MetaData' with another 'MetaB' instance
    meta.expand(MetaB(10))
