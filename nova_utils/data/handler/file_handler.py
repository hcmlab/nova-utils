""" Module for handling File data operations related to annotations and streams.

Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    18.8.2023

"""

import csv
import json
import math
import mmap
import subprocess
import xml.etree.ElementTree as Et
from pathlib import Path
from struct import *
from typing import Union

import decord
import ffmpegio
import numpy as np
from PIL import Image as PILImage
from decord import cpu

from nova_utils.data.annotation import (
    SchemeType,
    Annotation,
    DiscreteAnnotation,
    DiscreteAnnotationScheme,
    ContinuousAnnotation,
    ContinuousAnnotationScheme,
    FreeAnnotation,
    FreeAnnotationScheme,
)
from nova_utils.data.data import Data
from nova_utils.data.handler.ihandler import IHandler
from nova_utils.data.static import Image, Text
from nova_utils.data.stream import (
    Video,
    Audio,
    Stream,
    SSIStream,
    StreamMetaData,
    SSIStreamMetaData,
)
from nova_utils.utils.anno_utils import (
    convert_label_to_ssi_dtype,
    convert_ssi_to_label_dtype,
)
from nova_utils.utils.string_utils import string_to_enum
from nova_utils.utils.type_definitions import (
    SSILabelDType,
    SSIFileType,
    SSINPDataType,
)


# METADATA
class FileMetaData:
    """Metadata for a file.

    Attributes:
        file_path (Path): The filepath from which the data has been loaded
    """

    def __init__(self, file_path: Path):
        """
        Initialize metadata for any file.

        Args:
        file_path (Path): The filepath from which the data has been loaded
        """
        self.file_path = file_path


class FileSSIStreamMetaData:
    """Metadata for a file.

    Attributes:
        ftype (str): The file type.
        delim (str): The delimiter used to separate entries in the file.
    """

    def __init__(self, ftype: str, delim: str):
        """
        Initialize additional metadata for a file-based SSIStream.
        FileMetaData needs to be set separately.

        Args:
            ftype (str): The file type.
            delim (str): The delimiter used in the file.
        """
        self.ftype = ftype
        self.delim = delim


# ANNOTATIONS


class _AnnotationFileHandler(IHandler):
    """Class for handling the loading and saving of data annotations."""

    default_ext = '.annotation'

    @staticmethod
    def _load_data_discrete(path: Path, ftype: str):
        """
        Load discrete annotation data from a file.

        Args:
            path (Path): The path of the file containing the annotation data.
            ftype (str): The file type (ASCII or BINARY) of the annotation data.

        Returns:
            np.ndarray: The loaded discrete annotation data as a NumPy array.
        """

        if ftype == SSIFileType.ASCII.name:
            data = np.loadtxt(path, dtype=SSILabelDType.DISCRETE.value, delimiter=";", encoding='UTF-8')
        elif ftype == SSIFileType.BINARY.name:
            data = np.fromfile(path, dtype=SSILabelDType.DISCRETE.value)
        else:
            raise ValueError("FileType {} not supported".format(ftype))

        return data

    @staticmethod
    def _load_data_continuous(path, ftype):
        """
        Load continuous annotation data from a file.

        Args:
            path (Path): The path of the file containing the annotation data.
            ftype (str): The file type (ASCII or BINARY) of the annotation data.

        Returns:
            np.ndarray: The loaded continuous annotation data as a NumPy array.
        """
        if ftype == SSIFileType.ASCII.name:
            data = np.loadtxt(path, dtype=SSILabelDType.CONTINUOUS.value, delimiter=";", encoding='UTF-8')
        elif ftype == SSIFileType.BINARY.name:
            data = np.fromfile(path, dtype=SSILabelDType.CONTINUOUS.value)
        else:
            raise ValueError("FileType {} not supported".format(ftype))

        return data

    @staticmethod
    def _load_data_free(path, ftype, size):
        """
        Load free annotation data from a file.

        Args:
            path (Path): The path of the file containing the annotation data.
            ftype (str): The file type (ASCII or BINARY) of the annotation data.
            size (int): The size of the data to be loaded.

        Returns:
            np.ndarray: The loaded free annotation data as a NumPy array.
        """
        data = []
        if ftype == SSIFileType.ASCII.name:
            with open(path, "r") as ascii_file:
                ascii_file_reader = csv.reader(ascii_file, delimiter=";", quotechar='"', encoding='UTF-8')
                for row in ascii_file_reader:
                    f = float(row[0])
                    t = float(row[1])
                    n = row[2]
                    c = float(row[3])
                    data.append((f, t, n, c))

        elif ftype == SSIFileType.BINARY.name:
            with open(path, "rb") as binary_file:
                counter = 0
                binary_file.seek(0)

                while counter < size:
                    # from (8byte float)
                    f = unpack("d", binary_file.read(8))[0]
                    # to (8byte float)
                    t = unpack("d", binary_file.read(8))[0]
                    # length of label (4byte uint)
                    lol = unpack("i", binary_file.read(4))[0]
                    # the label (lol * byte)
                    n = binary_file.read(lol).decode("ISO-8859-1")
                    # confidence (4Byte float)
                    c = unpack("f", binary_file.read(4))[0]

                    data.append((f, t, n, c))
                    counter += 1
        else:
            raise ValueError("FileType {} not supported".format(ftype))

        return np.asarray(data, dtype=SSILabelDType.FREE.value)

    @staticmethod
    def _str_format_from_dtype(dtype: np.dtype):
        """
        Generate a string format for a given numpy dtype.

        Args:
            dtype (np.dtype): The numpy dtype.

        Returns:
            list: A list of format strings for each field in the dtype.
        """
        fmt = []

        for _, field_info in dtype.fields.items():
            dt, bo = field_info
            if np.issubdtype(dt, np.integer):
                # For integers, use '%d' format
                format_string = "%d"
            elif np.issubdtype(dt, np.floating):
                # For floating-point numbers, use '%.2f' format with 2 decimal places
                format_string = "%.2f"
            else:
                # For other data types (e.g., strings, bools, etc.), use the default '%s' format
                format_string = "%s"
            fmt.append(format_string)

        return fmt

    def load(self, fp: Path, header_only: bool = False) -> Annotation:
        """
        Load annotation data from an XML file.

        Args:
            fp (Path): The file path of the XML annotation file.
            header_only (bool): If true only the stream header will be loaded.


        Returns:
            Annotation: The loaded annotation data as an Annotation object.
        """
        data_path = fp.with_suffix(fp.suffix + "~")
        tree = Et.parse(fp)

        # info
        info = tree.find("info", {})
        ftype = info.get("ftype")
        size = int(info.get("size", 0))

        # meta
        meta = tree.find("meta", {})
        role = meta.get("role")
        annotator = meta.get("annotator")

        # scheme
        scheme = tree.find("scheme", {})
        scheme_name = scheme.get("name")
        scheme_type = scheme.get("type")

        # TODO: Nova Annotations do export a 'color' column where ssi annotations do not. Account for this
        # anno object
        # discrete scheme
        if scheme_type == SchemeType.DISCRETE.name:
            scheme_classes = {}
            for item in scheme:
                scheme_classes[item.get("id")] = item.get("name")

            anno_data = None
            if not header_only:
                anno_data = self._load_data_discrete(data_path, ftype)
                anno_data = convert_ssi_to_label_dtype(anno_data, SchemeType.DISCRETE)

            anno_scheme = DiscreteAnnotationScheme(
                name=scheme_name, classes=scheme_classes
            )
            annotation = DiscreteAnnotation(
                data=anno_data,
                scheme=anno_scheme,
                role=role,
                annotator=annotator,
            )

        # continuous scheme
        elif scheme_type == SchemeType.CONTINUOUS.name:
            sr = float(scheme.get("sr"))
            min_val = float(scheme.get("min"))
            max_val = float(scheme.get("max"))

            anno_data = None
            if not header_only:
                anno_data = self._load_data_continuous(data_path, ftype)
                anno_data = convert_ssi_to_label_dtype(anno_data, SchemeType.CONTINUOUS)
            anno_scheme = ContinuousAnnotationScheme(
                name=scheme_name, sample_rate=sr, min_val=min_val, max_val=max_val
            )
            annotation = ContinuousAnnotation(
                scheme=anno_scheme,
                data=anno_data,
                role=role,
                annotator=annotator,
            )

        # free scheme
        elif scheme_type == SchemeType.FREE.name:
            anno_data = None
            if not header_only:
                anno_data = self._load_data_free(data_path, ftype, size)
                anno_data = convert_ssi_to_label_dtype(anno_data, SchemeType.FREE)
            anno_scheme = FreeAnnotationScheme(name=scheme_name)
            annotation = FreeAnnotation(
                scheme=anno_scheme,
                data=anno_data,
                role=role,
                annotator=annotator,
            )
        else:
            raise TypeError(f"Unknown scheme type {type}")

        return annotation

    def save(self, data: Annotation, fp: Path, ftype: SSIFileType = SSIFileType.ASCII):
        """
        Save annotation data to a file.

        Args:
            data (Annotation): The annotation data to be saved.
            fp (Path): The file path for saving the data.
            ftype (SSIFileTypes, optional): The file type (ASCII or BINARY) for saving.

        Raises:
            TypeError: If filetype is not supported for saving or unknown
        """
        data_path = fp.with_suffix(fp.suffix + "~")

        # header
        root = Et.Element("annotation", attrib={"ssi-v ": "3"})

        # info
        size = str(len(data.data))
        Et.SubElement(root, "info", attrib={"ftype": ftype.name, "size": size})

        # meta
        role = "" if data.meta_data.role is None else data.meta_data.role
        annotator = "" if data.meta_data.annotator is None else data.meta_data.annotator
        Et.SubElement(root, "meta", attrib={"role": role, "annotator": annotator})

        # scheme
        scheme_name = data.annotation_scheme.name
        scheme_type = data.annotation_scheme.scheme_type

        if scheme_type == SchemeType.DISCRETE:
            data: DiscreteAnnotation
            scheme = Et.SubElement(
                root, "scheme", attrib={"name": scheme_name, "type": scheme_type.name}
            )
            for class_id, class_name in data.annotation_scheme.classes.items():
                Et.SubElement(
                    scheme, "item", attrib={"name": class_name, "id": str(class_id)}
                )

        elif scheme_type == SchemeType.CONTINUOUS:
            data: ContinuousAnnotation
            Et.SubElement(
                root,
                "scheme",
                attrib={
                    "name": scheme_name,
                    "type": scheme_type.name,
                    "sr": f"{data.annotation_scheme.sample_rate:.3f}",
                    "min": f"{data.annotation_scheme.min_val:.3f}",
                    "max": f"{data.annotation_scheme.max_val:.3f}",
                },
            )

        elif scheme_type == SchemeType.FREE:
            if ftype == SSIFileType.BINARY:
                raise TypeError(
                    "Binary output format is not supported for free annotation schemes"
                )
            data: FreeAnnotation
            Et.SubElement(
                root, "scheme", attrib={"name": scheme_name, "type": scheme_type.name}
            )
        else:
            raise TypeError(f"Unknown scheme type {type}")

        root = Et.ElementTree(root)
        Et.indent(root, space="    ", level=0)
        root.write(fp)

        anno_data = convert_label_to_ssi_dtype(data.data, scheme_type)

        # save data
        if ftype == SSIFileType.ASCII:
            fmt = self._str_format_from_dtype(anno_data.dtype)
            np.savetxt(data_path, anno_data, fmt=fmt, delimiter=";", encoding='UTF-8')
        if ftype == SSIFileType.BINARY:
            data.data.tofile(data_path, sep="")


# Text
class _TextFileHandler(IHandler):

    default_ext = '.txt'

    def load(self, fp, header_only=False) -> Union[Data, None]:
        text = np.loadtxt(fp)
        text = Text(data=text)
        return text

    def save(self, data, fp, header_only=False):
       with open(fp, 'w') as f:
           f.write(' '.join(data.data))


# Image
class _ImageFileHandler(IHandler):

    default_ext = '.jpg'

    def load(self, fp, header_only=False) -> Union[Data, None]:
        pil_img = PILImage.open(fp)
        pil_img = pil_img.convert('RGB')
        np_img = np.array(pil_img)

        img = Image(data=np_img, ext=pil_img.format, sample_shape=np_img.shape, dtype=np_img.dtype)
        return img

    def save(self, data, fp, header_only=False):
        np_img = data.data
        pil_img = PILImage.fromarray(np_img, mode="RGB")
        pil_img.save(fp)


# SSI STREAMS
class _SSIStreamFileHandler(IHandler):
    """Class for handling the loading and saving of SSIStreams."""

    default_ext = '.stream'

    def _load_header(self, fp: Path) -> dict:
        """
        Load SSIStream header from a file.

        Args:
            fp (Path): The file path of the SSIStream.

        Returns:
            dict: A dictionary containing SSIStream header data.
        """

        tree = Et.parse(fp)

        # info
        info = tree.find("info", {})
        ftype = info.get("ftype")
        sr = info.get("sr")
        dim = info.get("dim")
        byte = info.get("byte")
        dtype = info.get("type")
        delim = info.get("delim")

        # chunks
        chunks = []
        for chunk in tree.findall("chunk"):
            from_ = chunk.get("from")
            to_ = chunk.get("to")
            byte_ = chunk.get("byte")
            num_ = chunk.get("num")
            chunks.append((from_, to_, byte_, num_))

        chunks = np.array(chunks, dtype=SSIStream.CHUNK_DTYPE)
        num_samples = int(sum(chunks["num"]))
        duration = int(num_samples / float(sr) * 1000)

        ssistream_meta_data = {
            "name": fp.stem,
            "ext": fp.suffix,
            "duration": duration,
            "sample_shape": (int(dim),),
            "num_samples": num_samples,
            "sample_rate": float(sr),
            "dtype": string_to_enum(SSINPDataType, dtype).value,
            "chunks": chunks,
            "fp": fp,
            "delim": delim,
            "ftype": ftype,
        }
        return ssistream_meta_data

    def _load_data(
        self,
        fp: Path,
        size: int,
        dim: int,
        ftype=SSIFileType.ASCII,
        dtype: np.dtype = SSINPDataType.FLOAT.value,
        delim=" ",
    ):
        """
        Load SSIStream data from a file.

        Args:
            fp (Path): The file path of the SSIStream data.
            size (int): The size of the data.
            dim (int): The dimension of the data.
            ftype (SSIFileTypes, optional): The file type (ASCII or BINARY) of the SSIStream data.
            dtype (np.dtype, optional): The data type.
            delim (str, optional): The delimiter used in the file.

        Returns:
            np.ndarray: The loaded SSIStream data as a NumPy array.

        Raises:
            ValueError: If the provided filetype is not supported
        """
        if ftype == SSIFileType.ASCII:
            return np.loadtxt(fp, dtype=dtype, delimiter=delim)
        elif ftype == SSIFileType.BINARY:
            stream = np.fromfile(fp, dtype=dtype).reshape(-1, dim)
            if size is not None and size != 0:
                assert stream.shape[0] == size
            return stream
        else:
            raise ValueError("FileType {} not supported".format(self))

    def save(
        self,
        data: SSIStream,
        fp: Path,
        ftype: SSIFileType = SSIFileType.BINARY,
        delim: str = " ",
    ):
        """
        Save SSIStream data to a file.

        Args:
            data (SSIStream): The SSIStream data to be saved.
            fp (Path): The file path for saving the data.
            ftype (SSIFileTypes, optional): The file type (ASCII or BINARY) for saving.
            delim (str, optional): The delimiter to be used in the file.
        """
        # save header
        data_path = fp.with_suffix(fp.suffix + "~")

        # header
        root = Et.Element("annotation", attrib={"ssi-v ": "2"})

        # info
        meta_data: StreamMetaData | SSIStreamMetaData = data.meta_data
        sr = meta_data.sample_rate
        dim = (
            meta_data.sample_shape[0]
            if not meta_data.sample_shape is None
            else data.data[0].shape[0]
        )
        byte = np.dtype(meta_data.dtype).itemsize
        dtype = SSINPDataType(meta_data.dtype).name
        Et.SubElement(
            root,
            "info",
            attrib={
                "ftype": ftype.name,
                "sr": f"{sr:.3f}",
                "dim": str(dim),
                "byte": str(byte),
                "type": dtype,
                "delim": delim,
            },
        )

        # meta
        Et.SubElement(
            root,
            "meta",
            attrib={"type": meta_data.media_type, **meta_data.custom_meta}
        )

        # chunks
        if meta_data.chunks is not None:
            for chunk in meta_data.chunks:
                Et.SubElement(
                    root,
                    "chunk",
                    attrib={
                        "from": f"{chunk['from']:.3f}",
                        "to": f"{chunk['to']:.3f}",
                        "byte": str(chunk["byte"]),
                        "num": str(chunk["num"]),
                    },
                )

        # saving
        root = Et.ElementTree(root)
        Et.indent(root, space="    ", level=0)
        root.write(fp)

        # save data
        if ftype == SSIFileType.ASCII:
            np.savetxt(data_path, data.data, delimiter=delim)
        if ftype == SSIFileType.BINARY:
            data.data.tofile(data_path)

    def load(self, fp, header_only: bool = False, **kwargs) -> Data:
        """
        Load SSIStream data from a file.

        Args:
            fp (Path): The file path of the SSIStream.
            header_only (bool): If true only the stream header will be loaded.

            **kwargs: Arbitrary keyword arguments.

        Returns:
            Data: The loaded SSIStream data.
        """
        data_path = fp.with_suffix(fp.suffix + "~")
        header = self._load_header(fp)
        duration = header.get("duration")
        sample_shape = header.get("sample_shape")
        num_samples = header.get("num_samples")
        sample_rate = header.get("sample_rate")
        dtype = header.get("dtype")
        chunks = header.get("chunks")
        delim = header["delim"]
        ftype = header["ftype"]
        name = header["name"]
        ext = header["ext"]

        data = None
        if not header_only:
            data = self._load_data(
                fp=data_path,
                size=num_samples,
                dtype=dtype,
                dim=sample_shape[0],
                delim=delim,
                ftype=SSIFileType[ftype],
            )

        ssi_stream = SSIStream(
            data=data,
            duration=duration,
            sample_shape=sample_shape,
            num_samples=num_samples,
            sample_rate=sample_rate,
            dtype=dtype,
            chunks=chunks,
            name=name,
            ext=ext,
        )
        ssi_stream.meta_data.expand(FileSSIStreamMetaData(delim=delim, ftype=ftype))
        return ssi_stream


# VIDEO
class _LazyArray(np.ndarray):
    """LazyArray class extending numpy.ndarray for video and audio loading."""

    def __new__(cls, decord_reader, shape: tuple, dtype: np.dtype):
        buffer = mmap.mmap(
            -1, dtype.itemsize * math.prod(shape), access=mmap.ACCESS_READ
        )
        obj = super().__new__(cls, shape, dtype=dtype, buffer=buffer)

        obj.decord_reader = decord_reader
        obj.start_idx = 0
        return obj

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = list(range(index.start, index.stop))
            return self.decord_reader.get_batch(indices).asnumpy()
        else:
            return np.squeeze(self.decord_reader.get_batch([index]).asnumpy())


class _VideoFileHandler(IHandler):
    """Class for handling the loading and saving of video data."""

    default_ext = '.mp4'

    def _get_video_meta(self, fp) -> dict:
        """
        Get video metadata using ffprobe.

        Args:
            fp (Path): The file path of the video.

        Returns:
            dict: A dictionary containing video metadata.
        """
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-print_format",
            "json",
            "-show_streams",
            str(fp.resolve()),
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        metadata = json.loads(result.stdout)
        return metadata

    def load(self, fp: Path, header_only: bool = False) -> Data:
        """
        Load video data from a file.

        Args:
            fp (Path): The file path of the video.
            header_only (bool): If true only the stream header will be loaded.

        Returns:
            Data: The loaded video data.

        Raises:
            FileNotFoundError: Error if the file does not exist

        """
        if not fp.exists():
            raise FileNotFoundError(fp)
        # meta information
        metadata = self._get_video_meta(fp)
        metadata = metadata["streams"][0]
        _width = metadata.get("width")
        _height = metadata.get("height")
        _sample_rate = metadata.get("avg_frame_rate")

        sample_shape = (1, _height, _width, 3)
        duration = int(float(metadata.get("duration")) * 1000)
        sample_rate = eval(_sample_rate) if _sample_rate is not None else None
        num_samples = int(metadata.get("nb_frames"))
        dtype = np.dtype(np.uint8)

        # file loading
        video_reader = decord.VideoReader(str(fp.resolve()), ctx=cpu(0))

        lazy_video_data = None
        if not header_only:
            lazy_video_data = _LazyArray(
                video_reader,
                shape=(num_samples,) + sample_shape[1:],
                dtype=dtype,
            )

        video_ = Video(
            data=lazy_video_data,
            name=fp.stem,
            ext=fp.suffix,
            duration=duration,
            sample_shape=sample_shape,
            num_samples=num_samples,
            sample_rate=sample_rate,
            dtype=dtype,
        )
        return video_

    def save(self, data: Video, fp: Path):
        """
        Save video data to a file.

        Args:
            data (Video): The video data to be saved.
            fp (Path): The file path for saving the data.
        """
        meta_data: StreamMetaData = data.meta_data
        # sample_rate = int(meta_data.sample_rate)
        sample_rate = meta_data.sample_rate
        file_path = str(fp.resolve())

        # TODO: when directly using data.data as a numpy array it is always filled with zeros.
        #  this is an issue with the current implementation of the numpy array buffer in _LazyArray()
        #  creating a list from a buffered array and the a numpy array again solves this problem, but is probably very slow and memory intensive.
        ffmpegio.video.write(
            file_path, sample_rate, np.asarray(list(data.data)), overwrite=True
        )


# AUDIO
class _AudioFileHandler(IHandler):
    """Class for handling the loading and saving of audio data."""

    default_ext = '.wav'

    def _get_audio_meta(self, fp: Path) -> dict:
        """
        Get audio metadata using ffprobe.

        Args:
            fp (Path): The file path of the audio.

        Returns:
            dict: A dictionary containing audio metadata.
        """
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-i",
            str(fp.resolve()),
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        metadata = json.loads(result.stdout)
        return metadata

    def load(self, fp: Path, header_only: bool = False) -> Data:
        """
        Load audio data from a file.

        Args:
            fp (Path): The file path of the audio.
            header_only (bool): If true only the stream header will be loaded.


        Returns:
            Data: The loaded audio data.
        """
        # meta information
        stream_meta_data = self._get_audio_meta(fp)

        metadata = stream_meta_data["streams"][0]
        _channels = metadata.get("channels")
        _sample_rate = int(metadata.get("sample_rate"))
        _duration = int(float(metadata.get("duration")) * 1000)
        _num_samples = round(_duration * _sample_rate)

        sample_shape = (1, None, _channels)
        duration = _duration
        sample_rate = _sample_rate
        dtype = np.dtype(np.float32)
        num_samples = _num_samples

        # file loading
        lazy_audio_data = None
        if not header_only:
            audio_reader = decord.AudioReader(str(fp.resolve()), ctx=cpu(0))
            lazy_audio_data = _LazyArray(
                audio_reader, shape=audio_reader.shape, dtype=dtype
            )

        audio_ = Audio(
            data=lazy_audio_data,
            duration=duration,
            name=fp.stem,
            ext=fp.suffix,
            sample_shape=sample_shape,
            num_samples=num_samples,
            sample_rate=sample_rate,
            dtype=dtype,
        )
        return audio_

    def save(self, data: Audio, fp: Path):
        """
        Save audio data to a file.

        Args:
            data (Audio): The audio data to be saved.
            fp (Path): The file path for saving the data.
        """
        meta_data: StreamMetaData = data.meta_data
        ffmpegio.audio.write(
            str(fp.resolve()),
            int(meta_data.sample_rate),
            np.swapaxes(np.hstack(data.data), 0, -1),
            overwrite=True,
        )


class FileHandler(IHandler):
    """Class for handling different types of data files."""

    def _get_handler_for_fp(
        self, fp: Path
    ) -> Union[_AnnotationFileHandler , _SSIStreamFileHandler , _AudioFileHandler , _VideoFileHandler , _ImageFileHandler , _TextFileHandler]:
        """
        Get the appropriate handler for a given file.

        Args:
            fp (Path): The file path.

        Returns:
            IHandler: An instance of the appropriate data handler.
        """



        if not self.data_type:
            ext = fp.suffix[1:]
            if ext in ["annotation"]:
                return _AnnotationFileHandler()
            elif ext in ["stream"]:
                return _SSIStreamFileHandler()
            elif ext in ["wav", "mp3"]:
                return _AudioFileHandler()
            elif ext in ["mp4"]:
                return _VideoFileHandler()
            elif ext in [x[1:] for x in PILImage.registered_extensions()]:
                return _ImageFileHandler()
            elif ext in ['txt']:
                return _TextFileHandler()
            else:
                raise ValueError(f"Unsupported file extension {fp.suffix}")
        else:
            # TODO provide option to load data with unknown extensions by specifying the datatype
            raise NotImplementedError

    def _get_handler_for_dtype(self, dtype):
        if dtype == Text:
            return _TextFileHandler()
        elif dtype == Image:
            return _ImageFileHandler()
        elif dtype == Video:
            return  _VideoFileHandler()
        elif dtype == SSIStream:
            return  _SSIStreamFileHandler()
        elif dtype == Audio:
            return  _AudioFileHandler()
        elif dtype == DiscreteAnnotation or dtype == ContinuousAnnotation or dtype == FreeAnnotation:
            return  _AnnotationFileHandler()
        raise NotImplementedError

    def _get_handler(self, fp: Path =None, dtype=None):

        # Prefer dtype if passed
        if dtype:
            return self._get_handler_for_dtype(dtype)
        # Try to infer handler form extension
        elif fp:
            return self._get_handler_for_fp(fp)
        return None

    def __init__(self, data_type: int = None):
        self.data_type = data_type

    def load(self, fp: Union[Path, str], header_only: bool = False, dtype = None) -> Data:
        """
        Load data from a file.

        Args:
            fp (Path): The file path.
            header_only (bool): If true only the stream header will be loaded.

        Returns:
            Data: The loaded data.
        """
        if isinstance(fp, str):
            fp = Path(fp)
        handler = self._get_handler(fp, dtype)
        data = handler.load(fp, header_only=header_only)
        data.meta_data.name = Path(fp).name
        data.meta_data.expand(FileMetaData(fp))
        return data

    def save(self, data: Stream, fp: Union[Path, str], overwrite: bool = True, dtype = None, *args, **kwargs):
        """
        Save data to a file.

        Args:
            data (Union[Annotation, SSIStream, Video, Audio]): The data to be saved.
            fp (Path): The file path for saving the data.
            overwrite (bool, optional): Whether to overwrite the file if it exists.
            *args: Variable length argument list.
             **kwargs: Arbitrary keyword arguments.

        Raises:
            FileExistsError: If the file already exists and overwrite is not allowed.
        """
        if isinstance(fp, str):
            fp = Path(fp)
        handler = self._get_handler(fp, dtype)

        if not fp.suffix:
            fp = fp.parent / (fp.name + handler.default_ext)
        if fp.exists() and not overwrite:
            raise FileExistsError(f"Cannot write {fp} because file already exists")
        return handler.save(data, fp)


if __name__ == "__main__":
    # Test cases...
    test_annotations = False
    test_streams = False
    test_static = True
    base_dir = Path("../../../test_files/")
    fh = FileHandler()

    """TESTCASE FOR ANNOTATIONS"""
    if test_annotations:

        # ascii read
        discrete_anno_ascii = fh.load(base_dir / "discrete_ascii.annotation")
        continuous_anno_ascii = fh.load(base_dir / "continuous_ascii.annotation")
        free_anno_ascii = fh.load(base_dir / "free_ascii.annotation")

        # binary read
        discrete_anno_binary = fh.load(base_dir / "discrete_binary.annotation")
        continuous_anno_binary = fh.load(base_dir / "continuous_binary.annotation")

        # ascii write
        fh.save(discrete_anno_ascii, base_dir / "new_discrete_ascii.annotation")
        fh.save(continuous_anno_ascii, base_dir / "new_continuous_ascii.annotation")
        fh.save(free_anno_ascii, base_dir / "new_free_ascii.annotation")

        # binary write
        fh.save(
            discrete_anno_binary,
            base_dir / "new_discrete_binary.annotation",
            ftype=SSIFileType.BINARY,
        )
        fh.save(
            continuous_anno_binary,
            base_dir / "new_continuous_binary.annotation",
            ftype=SSIFileType.BINARY,
        )

        # verify
        discrete_anno_ascii_new = fh.load(base_dir / "new_discrete_ascii.annotation")
        continuous_anno_ascii_new = fh.load(
            base_dir / "new_continuous_ascii.annotation"
        )
        free_anno_ascii_new = fh.load(base_dir / "new_free_ascii.annotation")

        # binary read
        discrete_anno_binary_new = fh.load(base_dir / "new_discrete_binary.annotation")
        continuous_anno_binary_new = fh.load(
            base_dir / "new_continuous_binary.annotation"
        )

    """TESTCASE FOR STREAMS"""
    if test_streams:

        # ssistream read
        ssistream_ascii = fh.load(base_dir / "ascii.stream")
        ssistream_binary = fh.load(base_dir / "binary.stream")

        # Replace one dimension with random data
        new_data = ssistream_binary.data.copy()
        replacement_dimension = 0
        random_data = np.random.rand(new_data.shape[replacement_dimension])

        # Generate random data
        new_data[:, replacement_dimension] = random_data
        ssistream_binary.data = new_data
        ssistream_ascii.data = new_data

        # ssistream write
        fh.save(ssistream_ascii, base_dir / "new_ascii.stream", SSIFileType.ASCII)
        fh.save(ssistream_binary, base_dir / "new_binary.stream", SSIFileType.BINARY)

        # audio
        audio = fh.load(base_dir / "test_audio.wav")

        fh.save(audio, base_dir / "new_test_audio.wav")

        new_audio = fh.load(base_dir / "new_test_audio.wav")

        np.allclose(audio.data[0:10000], new_audio.data[0:10000])

        # video
        video = fh.load(base_dir / "test_video.mp4")

        fh.save(video, base_dir / "new_test_video.mp4")

        new_video = fh.load(base_dir / "new_test_video.mp4")

        assert new_video.data[0:30].all() == video.data[0:30].all()
        breakpoint()

    """TESTCASE FOR STATIC DATA"""
    if test_static:
        image = fh.load(base_dir / "test_image.png")
        image.data = image.data[..., [2,1,0]]
        fh.save(image, fp=(base_dir / "test_image_bgr.png"))