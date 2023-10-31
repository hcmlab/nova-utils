""" Module for handling File data operations related to annotations and streams.

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    24.10.2023

"""
import numpy as np
from pathlib import Path
import string
import random
from nova_utils.data.handler.file_handler import FileHandler
from nova_utils.data.static import StaticData, Text, Image
from nova_utils.data.stream import Stream
from nova_utils.data.annotation import Annotation
from nova_utils.data.handler.ihandler import IHandler
from nova_utils.data.data import Data
from nova_utils.utils.request_utils import DType

class RequestHandler(IHandler):
    """Class for handling user input"""

    def load(
        self, data, dtype, dataset: str = None, role: str = None, session: str = None
    ) -> Text:
        """
        Decode data received from server.

        Args:
        Returns:
            Data: The loaded data.
        """
        if dtype == Text:
            if isinstance(data, str):
                data = [data]
            return Text(np.asarray(data))
        elif dtype == Image:
            # TODO know decoding
            print("Don't know decoding. Lol!")
        else:
            raise ValueError(
                f"Data with unsupported dtype {dtype} received in request form."
            )

    def save(self, data: Data, shared_dir, job_id):
        """
        Save data to filesystem using the shared directory as well the current job id
        """

        # Create output folder for job
        output_dir = Path(shared_dir) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            output_name = data.meta_data.name
        except:
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