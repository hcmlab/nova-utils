import xml.etree.ElementTree as Et
import numpy as np
import csv
import decord
import subprocess
import json
from typing import Union
from decord import cpu
from struct import *
from nova_utils.data.idata import IData
from nova_utils.data.ssi_data_types import FileTypes, NPDataTypes, string_to_enum
from pathlib import Path
from nova_utils.data.data_handler.ihandler import IHandler
from nova_utils.data.annotation import (
    LabelType,
    SchemeType,
    Annotation,
    DiscreteAnnotation,
    DiscreteAnnotationScheme,
    ContinuousAnnotation,
    ContinuousAnnotationScheme,
    FreeAnnotation,
    FreeAnnotationScheme,
)
from nova_utils.data.stream import (
    Video,
    Audio,
    SSIStream,
    StreamMetaData,
    SSIStreamMetaData,

)
import mmap
import ffmpegio

# METADATA
class FileMetaData:
    def __init__(self, file_path):
        self.file_path = file_path


class FileSSIStreamMetaData(FileMetaData):
    def __init__(self, ftype: str, delim: str, **kwargs):
        super().__init__(**kwargs)
        self.ftype = ftype
        self.delim = delim


# ANNOTATIONS
class _AnnotationFileHandler(IHandler):
    """Class for handling the loading and saving of data annotations."""

    @staticmethod
    def _load_data_discrete(path, ftype):
        """
        Load discrete annotation data from a file.

        Parameters:
            path: The path of the file containing the annotation data.
            ftype: The file type (ASCII or BINARY) of the annotation data.

        Returns:
            numpy.ndarray: The loaded discrete annotation data as a NumPy array.
        """
        dt = LabelType.DISCRETE.value
        if ftype == FileTypes.ASCII.name:
            return np.loadtxt(path, dtype=dt, delimiter=";")
        elif ftype == FileTypes.BINARY.name:
            return np.fromfile(path, dtype=dt)
        else:
            raise ValueError("FileType {} not supported".format(ftype))

    @staticmethod
    def _load_data_continuous(path, ftype):
        """
        Load continuous annotation data from a file.

        Parameters:
            path: The path of the file containing the annotation data.
            ftype: The file type (ASCII or BINARY) of the annotation data.

        Returns:
            numpy.ndarray: The loaded continuous annotation data as a NumPy array.
        """
        dt = LabelType.CONTINUOUS.value
        if ftype == FileTypes.ASCII.name:
            return np.loadtxt(path, dtype=dt, delimiter=";")
        elif ftype == FileTypes.BINARY.name:
            return np.fromfile(path, dtype=dt)
        else:
            raise ValueError("FileType {} not supported".format(ftype))

    @staticmethod
    def _load_data_free(path, ftype, size):
        """
        Load free annotation data from a file.

        Parameters:
            path: The path of the file containing the annotation data.
            ftype: The file type (ASCII or BINARY) of the annotation data.
            size: The size of the data to be loaded.

        Returns:
            numpy.ndarray: The loaded free annotation data as a NumPy array.
        """
        data = []
        if ftype == FileTypes.ASCII.name:
            with open(path, "r") as ascii_file:
                ascii_file_reader = csv.reader(ascii_file, delimiter=";", quotechar='"')
                for row in ascii_file_reader:
                    f = float(row[0])
                    t = float(row[1])
                    n = row[2]
                    c = float(row[3])
                    data.append((f, t, n, c))

        elif ftype == FileTypes.BINARY.name:
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

        return np.array(data, LabelType.FREE.value)

    @staticmethod
    def _str_format_from_dtype(dtype: np.dtype):
        """
        Generate a string format for a given numpy dtype.

        Parameters:
            dtype (numpy.dtype): The numpy dtype.

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

    def load(self, fp) -> Annotation:
        """
        Load annotation data from an XML file.

        Parameters:
            fp (Path): The file path of the XML annotation file.

        Returns:
            Annotation: The loaded annotation data as an IAnnotation object.
        """
        data_path = fp.with_suffix(fp.suffix + "~")
        tree = Et.parse(fp)

        # info
        info = tree.find("info")
        ftype = info.get("ftype")
        size = info.get("size")

        # meta
        meta = tree.find("meta")
        if meta:
            role = meta.get("role")
            annotator = meta.get("annotator")

        # scheme
        scheme = tree.find("scheme")
        scheme_name = scheme.get("name")
        scheme_type = scheme.get("type")

        # TODO: Nova Annotations do export a 'color' column where ssi annotations do not. Account for this
        # anno object
        # discrete scheme
        if scheme_type == SchemeType.DISCRETE.name:
            scheme_classes = {}
            for item in scheme:
                scheme_classes[item.get("id")] = item.get("name")
            anno_data = self._load_data_discrete(data_path, ftype)
            anno_scheme = DiscreteAnnotationScheme(
                name=scheme_name, classes=scheme_classes
            )
            annotation = DiscreteAnnotation(
                #role=role,
                #annotator=annotator,
                scheme=anno_scheme,
                data=anno_data,
            )

        # continuous scheme
        elif scheme_type == SchemeType.CONTINUOUS.name:
            sr = float(scheme.get("sr"))
            min_val = float(scheme.get("min"))
            max_val = float(scheme.get("max"))
            anno_data = self._load_data_continuous(data_path, ftype)
            anno_scheme = ContinuousAnnotationScheme(
                name=scheme_name, sample_rate=sr, min_val=min_val, max_val=max_val
            )
            annotation = ContinuousAnnotation(
                #role=role,
                #annotator=annotator,
                scheme=anno_scheme,
                data=anno_data,
            )

        # free scheme
        elif scheme_type == SchemeType.FREE.name:
            anno_data = self._load_data_free(data_path, ftype, size)
            anno_scheme = FreeAnnotationScheme(name=scheme_name)
            annotation = FreeAnnotation(
                #role=role,
                #annotator=annotator,
                scheme=anno_scheme,
                data=anno_data,
            )
        else:
            raise TypeError(f"Unknown scheme type {type}")

        return annotation

    def save(self, data: Annotation, fp: Path, ftype: FileTypes = FileTypes.ASCII):

        data_path = fp.with_suffix(fp.suffix + "~")

        # header
        root = Et.Element("annotation", attrib={"ssi-v ": "3"})

        # info
        size = str(len(data.data))
        Et.SubElement(root, "info", attrib={"ftype": ftype.name, "size": size})

        # meta
        #TODO include meta again when implemented
        #role = data.info.meta_data.role
        #annotator = data.info.meta_data.annotator
        #Et.SubElement(root, "meta", attrib={"role": role, "annotator": annotator})

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
                    scheme, "item", attrib={"name": class_name, "id": class_id}
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
            if ftype == FileTypes.BINARY:
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

        # save data
        if ftype == FileTypes.ASCII:
            fmt = self._str_format_from_dtype(data.annotation_scheme.label_dtype)
            np.savetxt(data_path, data.data, fmt=fmt, delimiter=";")
        if ftype == FileTypes.BINARY:
            data.data.tofile(data_path, sep="")


# SSI STREAMS
class _SSIStreamFileHandler(IHandler):
    def _load_header(self, fp) -> dict:
        tree = Et.parse(fp)

        # info
        info = tree.find("info")
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

        chunks = np.array(chunks, dtype=SSIStreamMetaData.CHUNK_DTYPE)
        num_samples = int(sum(chunks["num"]))
        duration = num_samples * float(sr)

        ssistream_meta_data = {
            "duration":duration,
            "sample_shape":(int(dim),),
            "num_samples":num_samples,
            "sample_rate":float(sr),
            "dtype":string_to_enum(NPDataTypes, dtype).value,
            "chunks":chunks,
            "fp": fp,
            "delim": delim,
            "ftype": ftype
        }
        #file_metadata = FileSSIStreamMetaData(file_path=fp, delim=delim, ftype=ftype)

        return ssistream_meta_data

    def _load_data(
        self,
        fp: Path,
        size: int,
        dim: int,
        ftype=FileTypes.ASCII,
        dtype: np.dtype = NPDataTypes.FLOAT.value,
        delim=" ",
    ):
        if ftype == FileTypes.ASCII:
            return np.loadtxt(fp, dtype=dtype, delimiter=delim)
        elif ftype == FileTypes.BINARY:
            return np.fromfile(fp, dtype=dtype).reshape(size, dim)
        else:
            raise ValueError("FileType {} not supported".format(self))

    def save(
        self,
        data: SSIStream,
        fp: Path,
        ftype: FileTypes = FileTypes.BINARY,
        delim: str = " ",
    ):

        # save header
        data_path = fp.with_suffix(fp.suffix + "~")

        # header
        root = Et.Element("annotation", attrib={"ssi-v ": "2"})

        # info
        #meta_data: SSIStreamMetaData = data.info.meta_data
        sr = data.sample_rate
        dim = data.sample_shape[0]
        byte = np.dtype(data.dtype).itemsize
        dtype = NPDataTypes(data.dtype).name
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
        Et.SubElement(root, "meta")

        # chunks
        for chunk in data.chunks:
            Et.SubElement(
                root,
                "chunk",
                attrib={
                    "fromm": f"{chunk['from']:.3f}",
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
        if ftype == FileTypes.ASCII:
            np.savetxt(data_path, data.data, delimiter=delim)
        if ftype == FileTypes.BINARY:
            data.data.tofile(data_path)

    def load(self, fp, **kwargs) -> IData:
        data_path = fp.with_suffix(fp.suffix + "~")

        ssistream_meta_data = self._load_header(fp)

        ssistream_data = self._load_data(
            fp=data_path,
            size=ssistream_meta_data['num_samples'],
            dim=ssistream_meta_data['sample_shape'][0],
            ftype=FileTypes[ssistream_meta_data['ftype']],
            dtype=ssistream_meta_data['dtype'],
            delim=ssistream_meta_data['delim'],
        )

        #info = Info(handler=file_handler_meta_data)
        #ssi_stream = SSIStream(**ssistream_meta_data, data=ssistream_data, info=info)
        duration = ssistream_meta_data.get('duration')
        sample_shape = ssistream_meta_data.get('sample_shape')
        num_samples = ssistream_meta_data.get('num_samples')
        sample_rate = ssistream_meta_data.get('sample_rate')
        dtype = ssistream_meta_data.get('dtype')
        chunks = ssistream_meta_data.get('chunks')
        ssi_stream = SSIStream(data=ssistream_data, duration=duration, sample_shape=sample_shape, num_samples=num_samples, sample_rate=sample_rate, dtype = dtype, chunks=chunks)
        return ssi_stream


# VIDEO
class _LazyArray(np.ndarray):
    def __new__(cls, decord_reader, shape: tuple, dtype: np.dtype):
        buffer = mmap.mmap(-1, dtype.itemsize * np.prod(shape), access=mmap.ACCESS_READ)
        obj = super().__new__(cls, shape, dtype=dtype, buffer=buffer)

        obj.decord_reader = decord_reader
        obj.start_idx = 0
        return obj

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = list(range(index.start, index.stop))
            return self.decord_reader.get_batch(indices).asnumpy()
        else:
            return self.decord_reader.get_batch([index]).asnumpy()


class _VideoFileHandler(IHandler):
    def _get_video_meta(self, fp) -> dict:
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-print_format",
            "json",
            "-show_streams",
            fp,
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        metadata = json.loads(result.stdout)
        return metadata


    def load(self, fp: Path) -> IData:
        file_path = str(fp.resolve())

        # meta information
        metadata = self._get_video_meta(file_path)
        metadata = metadata["streams"][0]
        _width = metadata.get("width")
        _height = metadata.get("height")
        _sample_rate = metadata.get("avg_frame_rate")

        sample_shape = (1, _height, _width, 3)
        duration = float(metadata.get("duration"))
        sample_rate = (
            eval(_sample_rate) if _sample_rate is not None else None
        )
        num_samples = int(metadata.get("nb_frames"))
        dtype = np.dtype(np.uint8)

        # file loading
        video_reader = decord.VideoReader(file_path, ctx=cpu(0))
        lazy_video_data = _LazyArray(
            video_reader,
            shape=(num_samples,) + sample_shape[1:],
            dtype=dtype,
        )

        video_ = Video(data=lazy_video_data, duration=duration, sample_shape=sample_shape, num_samples=num_samples, sample_rate=sample_rate, dtype=dtype)
        return video_

    def save(self, data: Video, fp: Path):
        sr = data.sample_rate

        ffmpegio.video.write(
            str(fp.resolve()),
            int(sr),
            np.vstack(data.data),
            overwrite=True
        )


# AUDIO
class _AudioFileHandler(IHandler):
    def _get_audio_meta(self, fp) -> dict:
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-i",
            fp,
        ]
        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        metadata = json.loads(result.stdout)
        return metadata

    def load(self, fp: Path) -> IData:

        file_path = str(fp.resolve())

        # meta information
        stream_meta_data = self._get_audio_meta(file_path)

        metadata = stream_meta_data["streams"][0]
        _channels = metadata.get("channels")
        _sample_rate = int(metadata.get("sample_rate"))
        _duration = float(metadata.get("duration"))
        _num_samples = round(_duration * _sample_rate)

        sample_shape = (1, None, _channels)
        duration = _duration
        sample_rate = _sample_rate
        dtype = np.dtype(np.float32)
        num_samples = _num_samples

        # file loading
        audio_reader = decord.AudioReader(file_path, ctx=cpu(0))
        lazy_audio_data = _LazyArray(
            audio_reader, shape=audio_reader.shape, dtype=dtype
        )

        audio_ = Audio(data=lazy_audio_data, duration=duration, sample_shape=sample_shape, num_samples=num_samples, sample_rate=sample_rate, dtype=dtype)
        return audio_

    def save(self, data: Audio, fp: Path):
        sr = data.sample_rate
        ffmpegio.audio.write(
            str(fp.resolve()),
            int(sr),
            np.swapaxes(np.hstack(audio.data), 0, -1),
            overwrite=True,
        )


class FileHandler(IHandler):
    def _get_handler_for_file(
        self, fp
    ) -> Union[
        _AnnotationFileHandler,
        _SSIStreamFileHandler,
        _AudioFileHandler,
        _VideoFileHandler,
    ]:
        if not self.data_type:
            ext = fp.suffix[1:]
            if ext == "annotation":
                return _AnnotationFileHandler()
            elif ext == "stream":
                return _SSIStreamFileHandler()
            elif ext == "wav":
                return _AudioFileHandler()
            elif ext == "mp4":
                return _VideoFileHandler()
            else:
                raise ValueError(f"Unsupported file extension {fp.suffix}")
        else:
            # TODO provide option to load data with unknown extensions by specifying the datatype
            raise NotImplementedError

    def __init__(self, data_type: int = None):
        self.data_type = data_type

    def load(self, fp: Path) -> IData:
        handler = self._get_handler_for_file(fp)
        data = handler.load(fp)
        return data

    def save(self, data: any, fp: Path, *args, **kwargs):
        handler = self._get_handler_for_file(fp)
        return handler.save(data, fp, *args, **kwargs)


if __name__ == "__main__":

    test_annotations = True
    test_streams = False
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
            ftype=FileTypes.BINARY,
        )
        fh.save(
            continuous_anno_binary,
            base_dir / "new_continuous_binary.annotation",
            ftype=FileTypes.BINARY,
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
        random_data = np.random.rand(
            new_data.shape[replacement_dimension]
        )

        # Generate random data
        new_data[:, replacement_dimension] = random_data
        ssistream_binary.data = new_data
        ssistream_ascii.data = new_data

        # ssistream write
        fh.save(ssistream_ascii, base_dir / "new_ascii.stream", FileTypes.ASCII)
        fh.save(ssistream_binary, base_dir / "new_binary.stream", FileTypes.BINARY)

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