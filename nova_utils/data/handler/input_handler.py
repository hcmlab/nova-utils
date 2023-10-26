""" Module for handling File data operations related to annotations and streams.

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    24.10.2023

"""
import numpy as np
from nova_utils.data.static import Text
from pathlib import Path
from nova_utils.data.handler.ihandler import IHandler
from nova_utils.utils.type_definitions import (
    SSIFileType,
)
from nova_utils.utils.cache_utils import retreive_from_url
from typing import Union

class InputHandler(IHandler):
    """Class for handling user input"""

    def load(self, input_str: Union[str, list], dataset: str = None, role: str = None, session: str = None) -> Text:
        """
        Load data from a file.

        Args:
        Returns:
            Data: The loaded data.
        """
        if isinstance(input_str, str):
            input_str = [input_str]
        text_object = Text(data = np.asarray(input_str), dataset=dataset, role=role, session=session)
        return text_object


    def save(self, *args, **kwargs):
        raise NotImplementedError


if __name__ == "__main__":
   text = 'this is a test text'
   text_object = InputHandler().load(text, 'dataset', 'role','session')
   breakpoint()