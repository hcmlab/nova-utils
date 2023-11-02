""" Module for handling File data operations related to annotations and streams.

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    24.10.2023

"""
import random
import string
from pathlib import Path
from typing import Union

import numpy as np

from nova_utils.data.data import Data
from nova_utils.data.handler.file_handler import FileHandler
from nova_utils.data.handler.ihandler import IHandler
from nova_utils.data.static import Text, Image
from nova_utils.data.annotation import FreeAnnotation, FreeAnnotationScheme


class RequestHandler(IHandler):
    """Class for handling user input"""

    def load(
            self, data, dtype, dataset: str = None, role: str = None, session: str = None, header_only=False
    ) -> Union[Text, Image]:
        """
        Decode data received from server.

        Args:
        Returns:
            Data: The loaded data.
        """
        if dtype == Text:
            if header_only:
                return Text(data=None)
            if isinstance(data, str):
                data = [data]
            return Text(np.asarray(data))
        elif dtype == Image:
            if header_only:
                return Image(data=None)
            # TODO know decoding
            print("Don't know decoding. Lol!")
        elif dtype == FreeAnnotation:
            if header_only:
                return FreeAnnotation(data=None, scheme=FreeAnnotationScheme(name='default_scheme'))
        else:
            raise ValueError(
                f"Data with unsupported dtype {dtype} received in request form."
            )

    def save(self, data: Data, shared_dir, job_id, dataset=None, session=None):
        """
        Save data to filesystem using the shared directory as well the current job id
        """

        # Create output folder for job
        output_dir = Path(shared_dir) / job_id
        if dataset:
            output_dir /= dataset
        if session:
            output_dir /= session
        output_dir.mkdir(parents=True, exist_ok=True)

        if data.meta_data.name:
            output_name = data.meta_data.name
        else:
            output_name = "".join(
                random.choices(string.ascii_uppercase + string.digits, k=6)
            )

        handler = FileHandler()
        handler.save(data, output_dir / output_name, dtype=type(data))


if __name__ == "__main__":
    text = "this is a test text"
    text_object = RequestHandler().load(text, Text, "dataset", "role", "session")
    RequestHandler().save(text_object, '../../shared', 'my_job')
    breakpoint()
