import xml.etree.ElementTree as ET
import numpy as np
from nova_utils.ssi_utils.ssi_data_types import NPDataTypes, FileTypes, string_to_enum
from pathlib import Path

class Chunk:
    def __init__(self, f=0, t=0, b=0, n=0):
        self.f = f
        self.t = t
        self.b = b
        self.n = n


class Stream:
    def __init__(
        self,
        ftype: FileTypes = FileTypes.UNDEF,
        sr: int = 0,
        dim: int = 0,
        byte: int = 4,
        dtype: NPDataTypes = NPDataTypes.UNDEF,
        delim: str = "",
        chunks: list = None,
        data: np.ndarray = None,
        path: Path =None,
    ):
        self.ftype = ftype
        self.sr = sr
        self.dim = dim
        self.byte = byte
        self.dtype = dtype
        self.delim = delim
        self.chunks = chunks if chunks else []
        self.data = data

        if path:
            self.load(path)

    def load_header(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        chunks = 0

        for child in root:
            if child.tag == "info":
                for key, val in child.attrib.items():
                    if key == "ftype":
                        self.ftype = string_to_enum(FileTypes, val)
                    elif key == "sr":
                        self.sr = float(val)
                    elif key == "dim":
                        self.dim = int(val)
                    elif key == "byte":
                        self.byte = int(val)
                    elif key == "type":
                        self.dtype = string_to_enum(NPDataTypes, val).value
                    elif key == "delim":
                        self.delim = val
            elif child.tag == "chunk":
                f, t, b, n = 0, 0, 0, 0
                for key, val in child.attrib.items():
                    if key == "from":
                        f = float(val)
                    elif key == "to":
                        t = float(val)
                    elif key == "num":
                        n = int(val)
                    elif key == "byte":
                        b = int(val)
                chunks += 1
                self.chunks.append(Chunk(f, t, b, n))

    def load_data(self, path):
        if self.ftype == FileTypes.ASCII:
            self.data = np.loadtxt(path, dtype=self.dtype, delimiter=self.delim)
        elif self.ftype == FileTypes.BINARY:
            # number of all data chunks
            num = sum([c.n for c in self.chunks])
            self.data = np.fromfile(path, dtype=self.dtype).reshape(num, self.dim)
        else:
            raise ValueError("FileType {} not supported".format(self))

    def load(self, path: Path):
        if not path == type(Path):
            path = Path(path)
        self.load_header(path)
        self.load_data(path.parent / (path.name + '~'))

    def save_header(self, path):

        root = ET.Element("stream")
        ET.SubElement(
            root,
            "info",
            ftype=str(self.ftype.name),
            sr=str(self.sr),
            dim=str(self.dim),
            byte=str(self.byte),
            type=NPDataTypes(self.dtype).name,
            # delim=str(self.delim),
        )
        ET.SubElement(
            root,
            "meta",
        )

        for chunk in self.chunks:
            chunk = {
                "from": str(chunk.f),
                "to": str(chunk.t),
                "byte": str(chunk.b),
                "num": str(chunk.n),
            }

            ET.SubElement(root, "chunk", **chunk)

        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ", level=0)
        tree.write(path)

    def save_data(self, path):
        if self.ftype == FileTypes.ASCII:
            np.savetxt(path, self.data, delimiter=self.delim)
        if self.ftype == FileTypes.BINARY:
            self.data.tofile(path)

    def save(self, path: Path):
        if not path == type(Path):
            path = Path(path)
        if not path.suffix == '.stream':
            path = path.parent / (path.name + '.stream')
        self.save_header(path)
        self.save_data(path.parent / (path.name + '~'))


if __name__ == "__main__":
    stream = Stream()
    stream.load(Path("../../local/novice.audio.gemaps[480ms,40ms,480ms].stream"))

    # modify data
    stream.data = np.random.rand(sum([c.n for c in stream.chunks]), stream.dim).astype(
        np.float32
    )
    stream.ftype = FileTypes.ASCII

    stream.save(Path("../../local/novice.audio.gemaps[480ms,40ms,480ms]_modified.stream"))
