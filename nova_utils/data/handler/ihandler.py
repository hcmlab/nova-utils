""" Interface class for all data handlers

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    18.8.2023

"""

from abc import ABC, abstractmethod
from typing import Union

from nova_utils.data.data import Data


class IHandler(ABC):
    """
    Abstract base class for data handling operations.

    Subclasses of IHandler must implement the methods 'load' and 'save'
    to handle data loading and saving operations, respectively.
    """

    default_ext = '.data'

    @abstractmethod
    def load(self, *args, **kwargs) -> Union[Data, None]:
        """
        Load data using specified arguments and keyword arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Data: A Data object representing the loaded data.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, *args, **kwargs):
        """
        Save data using specified arguments and keyword arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError()
