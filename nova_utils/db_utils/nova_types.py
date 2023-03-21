from enum import Enum

class DataTypes(Enum):
  VIDEO = 0
  AUDIO  = 1
  FEATURE = 2


class AnnoTypes(Enum):
  DISCRETE = 0
  CONTINUOUS = 1
  FREE = 2
  POINT = 3
  DISCRETE_POLYGON = 4


'''Helper'''
def string_to_enum(enum, string):
  for e in enum:
    if e.name == string.upper():
      return e
  raise ValueError('{} not part of enumeration  {}'.format(string, enum))
