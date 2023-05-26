import xml.etree.ElementTree as ET
import numpy as np
from nova_utils.ssi_utils.ssi_data_types import FileTypes, string_to_enum
from nova_utils.ssi_utils.ssi_stream_utils import Stream
import os

class Sample:
    def __init__(self, streams = [], user_id = None, class_id = None, score = 0):
        self.streams = streams
        self.user_id = user_id
        self.class_id = class_id
        self.score = score

class SampleList:
    #def __init__(self, samples = None, size = 0,  missing = False, garbage=0, stream_paths = [], streams = [], classes = {}, user = {}, is_discrete = False):
    def __init__(self):
        self.samples = [] 
        self.samples = None
        self.size = 0
        self.missing = False
        self.garbage = 0
        self.stream_paths = []
        self.streams = []
        self.classes = {}
        self.user = {}
        self.is_discrete = False

    def load_header(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        for child in root:
            if child.tag ==  'info':
                for key,val in child.attrib.items():
                    if key == 'ftype':
                        self.ftype = string_to_enum(FileTypes, val)
                    elif key == 'size':
                        self.size = int(val)
                    elif key == 'missing':
                        self.missing = val
                    elif key == 'garbage':
                        self.garbage = int(val)
               
            elif child.tag == 'streams':
                for item in child:                    
                    for key,val in item.attrib.items():
                        if key == 'path':
                            self.stream_paths.append(val)              
            elif child.tag == 'classes':
                id = 0
                for item in child:
                    for key,val in item.attrib.items():
                        if key == "name":
                            class_name = val
                        elif key == "id":
                            id = val
                    self.classes[id] = class_name
                    id += 1
                if len(self.classes) > 0:
                    self.is_discrete = True
            elif child.tag == 'users':
                id = 0
                for item in child:
                    for key,val in item.attrib.items():
                        if key == "name":
                            user_name = val
                        elif key == "id":
                            id = val
                    self.user[id] = user_name
                    id += 1
        return self

    def load_samples(self, path):
        dt = {'names':('user_id', 'class_id', 'score', 'time'),
                          'formats':('i4', 'i4', 'f4', 'f8')}
        if self.ftype == FileTypes.ASCII:
            self.samples = np.loadtxt(path, dtype=dt,delimiter=" ")
        elif self.ftype == FileTypes.BINARY:
            self.samples = (np.fromfile(path, dtype=dt))
        else:
            raise ValueError('FileType {} not supported'.format(self))

    def load_streams(self, path):
        dir = os.path.dirname(path)
        
        for s in self.stream_paths:
            stream_path =  os.path.join(os.path. sep,os.getcwd(), dir, s + '.stream')
            stream = Stream().load(stream_path)
            self.streams.append(stream)

    def load(self, path):
        self.load_header(path)
        self.load_samples(path + '~')
        self.load_streams(path)
       
if __name__ == "__main__":
    sample_list_ascii = SampleList()
    sample_list_ascii.load("Testfiles/sl_ascii.samples")

    sample_list_binary = SampleList()
    sample_list_binary.load("Testfiles/sl_binary.samples")
