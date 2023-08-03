import xml.etree.ElementTree as Et
import numpy as np
import csv
from struct import *
from nova_utils.data.ssi_data_types import FileTypes
from pathlib import Path
from nova_utils.data.data_handler.ihandler import IHandler
from nova_utils.data.annotation import (
    LabelType,
    SchemeType,
    IAnnotation,
    DiscreteAnnotation,
    DiscreteAnnotationScheme,
    ContinuousAnnotation,
    ContinuousAnnotationScheme,
    FreeAnnotation,
    FreeAnnotationScheme,
)

class AnnotationFileHandler(IHandler):
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

    def load(self, fp) -> IAnnotation:
        """
        Load annotation data from an XML file.

        Parameters:
            fp (Path): The file path of the XML annotation file.

        Returns:
            IAnnotation: The loaded annotation data as an IAnnotation object.
        """
        data_path = fp.with_suffix(fp.suffix + "~")
        tree = Et.parse(fp)

        # info
        info = tree.find("info")
        ftype = info.get("ftype")
        size = info.get("size")

        # meta
        meta = tree.find("meta")
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
                role=role,
                annotator=annotator,
                annotation_scheme=anno_scheme,
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
                role=role,
                annotator=annotator,
                annotation_scheme=anno_scheme,
                data=anno_data,
            )

        # free scheme
        elif scheme_type == SchemeType.FREE.name:
            anno_data = self._load_data_free(data_path, ftype, size)
            anno_scheme = FreeAnnotationScheme(name=scheme_name)
            annotation = FreeAnnotation(
                role=role,
                annotator=annotator,
                annotation_scheme=anno_scheme,
                data=anno_data,
            )
        else:
            raise TypeError(f"Unknown scheme type {type}")

        return annotation

    def save(self, anno: IAnnotation, fp: Path, ftype=FileTypes.ASCII):

        data_path = fp.with_suffix(fp.suffix + "~")

        # header
        root = Et.Element("annotation", attrib={"ssi-v ": "3"})

        # info
        size = str(len(anno.data))
        Et.SubElement(root, "info", attrib={"ftype": ftype.name, "size": size})

        # meta
        role = anno.meta_info.role
        annotator = anno.meta_info.annotator
        Et.SubElement(root, "meta", attrib={"role": role, "annotator": annotator})

        # scheme
        scheme_name = anno.annotation_scheme.name
        scheme_type = anno.annotation_scheme.scheme_type

        if scheme_type == SchemeType.DISCRETE:
            anno: DiscreteAnnotation
            scheme = Et.SubElement(
                root, "scheme", attrib={"name": scheme_name, "type": scheme_type.name}
            )
            for class_id, class_name in anno.annotation_scheme.classes.items():
                Et.SubElement(
                    scheme, "item", attrib={"name": class_name, "id": class_id}
                )

        elif scheme_type == SchemeType.CONTINUOUS:
            anno: ContinuousAnnotation
            Et.SubElement(
                root,
                "scheme",
                attrib={
                    "name": scheme_name,
                    "type": scheme_type.name,
                    "sr": f"{anno.annotation_scheme.sample_rate:.3f}",
                    "min": f"{anno.annotation_scheme.min_val:.3f}",
                    "max": f"{anno.annotation_scheme.max_val:.3f}",
                }
            )

        elif scheme_type == SchemeType.FREE:
            if ftype == FileTypes.BINARY:
                raise TypeError("Binary output format is not supported for free annotation schemes")
            anno: FreeAnnotation
            Et.SubElement(
                root,
                "scheme",
                attrib={
                    "name" : scheme_name,
                    "type" : scheme_type.name
                }
            )
        else:
            raise TypeError(f"Unknown scheme type {type}")

        root = Et.ElementTree(root)
        Et.indent(root, space="    ", level=0)
        root.write(fp)

        # save data
        if ftype == FileTypes.ASCII:
            fmt = self._str_format_from_dtype(anno.annotation_scheme.label_dtype)
            np.savetxt(data_path, anno.data, fmt=fmt, delimiter=";")
        if ftype == FileTypes.BINARY:
            anno.data.tofile(data_path, sep="")


if __name__ == "__main__":

    """TESTCASE FOR ANNOTATIONS"""

    base_dir = Path("../../../test_files/")
    afh = AnnotationFileHandler()

    # ascii read
    discrete_anno_ascii = afh.load(base_dir / "discrete_ascii.annotation")
    continuous_anno_ascii = afh.load(base_dir / "continuous_ascii.annotation")
    free_anno_ascii = afh.load(base_dir / "free_ascii.annotation")

    # binary read
    discrete_anno_binary = afh.load(base_dir / "discrete_binary.annotation")
    continuous_anno_binary = afh.load(base_dir / "continuous_binary.annotation")

    # ascii write
    afh.save(discrete_anno_ascii, base_dir / "discrete_ascii_new.annotation")
    afh.save(continuous_anno_ascii, base_dir / "continuous_ascii_new.annotation")
    afh.save(free_anno_ascii, base_dir / "free_ascii_new.annotation")

    # binary write
    afh.save(discrete_anno_binary, base_dir / "discrete_binary_new.annotation", ftype=FileTypes.BINARY)
    afh.save(continuous_anno_binary, base_dir / "continuous_binary_new.annotation", ftype=FileTypes.BINARY)

    # verify
    discrete_anno_ascii_new = afh.load(base_dir / "discrete_ascii_new.annotation")
    continuous_anno_ascii_new = afh.load(base_dir / "continuous_ascii_new.annotation")
    free_anno_ascii_new = afh.load(base_dir / "free_ascii_new.annotation")

    # binary read
    discrete_anno_binary_new = afh.load(base_dir / "discrete_binary_new.annotation")
    continuous_anno_binary_new = afh.load(base_dir / "continuous_binary_new.annotation")

breakpoint()
