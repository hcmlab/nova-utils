from nova_utils.data.idata import IData
from abc import ABC, abstractmethod

class IHandler(ABC):
    """
    Abstract base class for data handlers.

    This class defines the interface for handling data loading and saving from a data source, such as Mongo db.

    Attributes:
        None

    Methods:
        load(*args, **kwargs) -> IData:
            Abstract method for loading data from the data source.

            This method must be implemented in the child classes to provide specific data loading functionality.

            Parameters:
                *args: Variable-length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                IData: An object representing the loaded data.

            Raises:
                NotImplementedError: This error is raised when the child class does not implement the load method.

        save(*args, **kwargs):
            Abstract method for saving data to the data source.

            This method must be implemented in the child classes to provide specific data saving functionality.

            Parameters:
                *args: Variable-length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                IData: An object representing the saved data.

            Raises:
                NotImplementedError: This error is raised when the child class does not implement the save method.
    """
    @abstractmethod
    def load(self, *args, **kwargs) -> IData:
        """
        Abstract method for loading data from the data source.

        This method must be implemented in the child classes to provide specific data loading functionality.

        Parameters:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            IData: An object representing the loaded data.

        Raises:
            NotImplementedError: This error is raised when the child class does not implement the load method.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, *args, **kwargs):
        """
        Abstract method for saving data to the data source.

        This method must be implemented in the child classes to provide specific data saving functionality.

        Parameters:
            *args: Variable-length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            IData: An object representing the saved data.

        Raises:
            NotImplementedError: This error is raised when the child class does not implement the save method.
        """
        raise NotImplementedError()
