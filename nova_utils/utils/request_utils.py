from enum import Enum

from nova_utils.data.annotation import DiscreteAnnotation, FreeAnnotation, ContinuousAnnotation
from nova_utils.data.data import Data
from nova_utils.data.static import Text, Image
from nova_utils.data.stream import SSIStream, Audio, Video


class Source(Enum):
    DB = "db"  # data.handler.NovaDBHandler
    FILE = "file"  # data.handler.FileHandler
    URL = "url"  # data.handler.UrlHandler
    REQUEST = "request"


class DType(Enum):
    STREAM = "stream"
    ANNO = "annotation"#"anno"
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"

def data_description_to_string(data_desc: dict) -> str:
    """
    Convert data description to a string representation.

    Args:
        data_desc (dict): Data description dictionary.

    Returns:
        str: String representation of the data description.
    """

    id = data_desc.get("id")
    if id is not None:
        return id

    src, type_ = data_desc["src"].split(":")
    delim = "_"
    if src == "db":
        if type_ == "anno":
            return delim.join(
                [data_desc["scheme"], data_desc["annotator"], data_desc["role"]]
            )
        elif type_ == "stream":
            return delim.join([data_desc["name"], data_desc["role"]])
        else:
            raise ValueError(f"Unknown data type {type_} for data.")
    elif src == "file":
        return delim.join([data_desc["fp"]])
    else:
        raise ValueError(f"Unsupported source type {src} for generating data description ids")


def parse_src(desc):
    try:
        src, dtype = desc["src"].split(":", 1)
        src = Source(src)
        dtype_specific = None
        if ':' in dtype:
            dtype, dtype_specific = dtype.split(':', 1)
        dtype = DType(dtype)
    except:
        raise ValueError(f'Invalid value for data source {desc["src"]}')
    return src, dtype, dtype_specific

def dtype_from_desc(desc: str):
    try:
        _, dtype = desc["src"].split(":", 1)
        dtype_specific = None
        if ':' in dtype:
            dtype, dtype_specific = dtype.split(':', 1)
        dtype = DType(dtype)

        if dtype == DType.STREAM:
            if dtype_specific == 'audio':
                return Audio
            elif dtype_specific == 'video':
                return  Video
            elif dtype_specific == 'ssistream':
                return SSIStream
            else:
                raise ValueError(f'Unknown annotation type {dtype_specific}')
        elif dtype == DType.ANNO:
            if dtype_specific == 'free':
                return FreeAnnotation
            elif dtype_specific == 'discrete':
                return  DiscreteAnnotation
            elif dtype_specific == 'continuous':
                return ContinuousAnnotation
            else:
                raise ValueError(f'Unknown annotation type {dtype_specific}')
        elif dtype == DType.TEXT:
            return Text
        elif dtype == DType.IMAGE:
            return Image
        else:
            raise ValueError(f'Unknown dtype {dtype}')

    except Exception as e:
        print(e)
    return Data
