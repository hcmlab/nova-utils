import xml.etree.ElementTree as ET
from nova_utils.ssi_utils.ssi_data_types import FileTypes, string_to_enum
from enum import Enum
import numpy as np
import csv
from struct import *


class SchemeType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    FREE = 2

class Scheme:
    def __init__(self, name : str ="", type: SchemeType = "", sr: float = 0, min: float = 0, max: float = 0, classes: dict = None ):
        self.name = name
        self.type = type
        self.sr = sr
        self.min = min
        self.max = max
        self.classes = classes if classes else {}

class Anno:
    ftype: FileTypes

    #def __init__(self, ftype="UNDEF", size=0, role="", annotator=""):
    def __init__(self,
                 ftype: FileTypes = FileTypes.UNDEF,
                 size: int = 0,
                 role: str = "",
                 annotator: str = "",
                 data: list = None,
                 scheme: Scheme = None
                 ):
        self.ftype = ftype
        self.size = size
        self.role = role
        self.annotator = annotator
        self.data = data
        self.scheme = scheme if scheme else Scheme()

    def load_header(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        for child in root:
            for key,val in child.attrib.items():
                if child.tag == 'info':  
                    if key == 'ftype':
                        self.ftype = string_to_enum(FileTypes, val)
                    elif key == 'size':
                        self.size = int(val)       
                elif child.tag == 'meta':
                    if key == 'role':
                        self.role = val
                    elif key == 'annotator':
                        self.annotator = val
                elif child.tag == 'scheme':
                    if key == 'name':
                        self.scheme.name = val
                    if key == 'type':
                        self.scheme.type = string_to_enum(SchemeType, val)
                    if key == 'sr':
                        self.scheme.sr = float(val)
                    if key == 'min':
                        self.scheme.min = int(val)
                    if key == 'max':
                        self.scheme.max = int(val)
            if child.tag == 'scheme' and self.scheme.type == SchemeType.DISCRETE :
                for item in child:              
                    for key,val in item.attrib.items():
                        if key == "name":
                            class_name = val
                        elif key == "id":
                            id = val
                    self.scheme.classes[id] = class_name

    def load_data_discrete(self, path):
        dt = {'names':('from', 'to', 'class', 'confidence'),
                          'formats':(np.float32, np.float32, np.int32, np.float32)}
        if self.ftype == FileTypes.ASCII:
            self.data = np.loadtxt(path, dtype=dt, delimiter=';')
        elif self.ftype == FileTypes.BINARY:
            self.data = np.fromfile(path, dtype=dt)
        else:
            raise ValueError('FileType {} not supported'.format(self.ftype))

    def load_data_continuous(self, path):
        dt = {'names':('score', 'confidence'),
                          'formats':(np.float32, np.float32)}
        if self.ftype == FileTypes.ASCII:
            self.data = np.loadtxt(path, dtype=dt, delimiter=';')
        elif self.ftype == FileTypes.BINARY:
            self.data = np.fromfile(path, dtype=dt)
        else:
            raise ValueError('FileType {} not supported'.format(self.ftype))

    def load_data_free(self, path):

        data = []
        dt = {'names':('from', 'to', 'name', 'confidence'),
              'formats':(np.float64, np.float64, np.object_, np.float32)}

        if self.ftype == FileTypes.ASCII:
            with open(path, "r") as ascii_file:
                ascii_file_reader = csv.reader(ascii_file, delimiter=';', quotechar='"')
                for row in ascii_file_reader:
                    f = float(row[0])
                    t = float(row[1])
                    n = row[2]
                    c = float(row[3])
                    data.append((f,t,n,c))

        elif self.ftype == FileTypes.BINARY:
            with open(path, "rb") as binary_file:
                counter = 0
                binary_file.seek(0)  
                
                while counter < self.size:
                    #from (8byte float)
                    f = unpack('d', binary_file.read(8))[0]
                    ##to (8byte float)
                    t = unpack('d', binary_file.read(8))[0]
                    #length of label (4byte uint)
                    lol = unpack('i', binary_file.read(4))[0]
                    #the label (lol * byte)
                    n = binary_file.read(lol).decode("ISO-8859-1")
                    #confidence (4Byte float)
                    c = unpack('f', binary_file.read(4))[0]

                    data.append((f,t,n,c))
                    counter += 1
        else:
            raise ValueError('FileType {} not supported'.format(self.ftype))

        self.data = np.array(data, dt)

    def load_data(self, path):
        if self.scheme.type == SchemeType.DISCRETE:
            self.load_data_discrete(path)
        elif self.scheme.type == SchemeType.CONTINUOUS:
            self.load_data_continuous(path)
        elif self.scheme.type == SchemeType.FREE:
            self.load_data_free(path)
        else:
            raise ValueError('SchemeType {} not supported'.format(self.scheme.type))
        return self.data

    def load(self, path):
        self.load_header(path)
        self.load_data(path + '~')
    
if __name__ == "__main__":

    '''discrete annotations'''
    anno_discrete_ascii = Anno()
    anno_discrete_ascii.load("Testfiles/discrete_ascii.annotation")

    anno_discrete_binary = Anno()
    anno_discrete_binary.load("Testfiles/discrete_binary.annotation")

    '''continous annotations'''
    anno_cont_ascii = Anno()
    anno_cont_ascii.load("Testfiles/continuous_ascii.annotation")

    anno_cont_binary = Anno()
    anno_cont_binary.load("Testfiles/continuous_binary.annotation")
    
    '''free annotations'''
    anno_free_ascii = Anno()
    anno_free_ascii.load("Testfiles/free_ascii.annotation")

    anno_free_binary = Anno()
    anno_free_binary.load("Testfiles/free_binary.annotation")
  
    ...
