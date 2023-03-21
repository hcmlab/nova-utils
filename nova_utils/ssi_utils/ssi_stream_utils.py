import xml.etree.ElementTree as ET
import numpy as np
from nova_utils.ssi_utils.ssi_data_types import NPDataTypes, FileTypes, string_to_enum

class Stream:
    #def __init__(self, ftype="UNDEF" , sr=0.0, dim=0, byte=4, type="UNDEF", delim="", chunks = []):
    def __init__(self, path=None):
        self.ftype = string_to_enum(FileTypes, "UNDEF")
        self.sr = 0
        self.dim = 0
        self.byte = 4
        self.type = "UNDEF"
        self.delim = ""
        self.chunks = []
        self.data = None

        if path:
            self.load(path)

    def load_header(self, path):
        tree = ET.parse(path)
        root = tree.getroot()
        chunks = 0

        for child in root:
            if child.tag ==  'info':
                for key,val in child.attrib.items():
                    if key == 'ftype':
                        self.ftype = string_to_enum(FileTypes, val)
                    elif key == 'sr':
                        self.sr = float(val)
                    elif key == 'dim':
                        self.dim = int(val)
                    elif key == 'byte':
                        self.byte = int(val)
                    elif key == 'type':
                        self.type = string_to_enum(NPDataTypes, val).value
                    elif key == 'delim':
                        self.delim = val
            elif child.tag == 'chunk':
                f, t, b, n = 0, 0, 0, 0
                for key,val in child.attrib.items():
                    if key == 'from':
                        f = float(val)
                    elif key == 'to':
                        t = float(val)
                    elif key == 'num':
                        n = int(val)
                    elif key == 'byte':
                        b = int(val)
                chunks += 1
                self.chunks.append( [f, t, b, n] )
        return self

    def load_data(self, path):
        if self.ftype == FileTypes.ASCII:
            self.data = np.loadtxt(path, dtype=self.type, delimiter=self.delim)
        elif self.ftype == FileTypes.BINARY:
            num = np.sum(self.chunks, axis=0, dtype=np.int32)[3]
            self.data = np.fromfile(path, dtype=self.type).reshape(num, self.dim)
        else:
            raise ValueError('FileType {} not supported'.format(self))
        return self.data

    def load(self, path):
        self.load_header(path)
        self.load_data(path + '~')
        return self

if __name__ == "__main__":
    stream_ascii = Stream().load("Testfiles/ascii.stream")
    stream_bin = Stream().load("Testfiles/binary.stream")
