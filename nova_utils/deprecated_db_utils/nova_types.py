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
  """

  Searches for 'string' in the enum names and returns the corresponding enum object.
  If the passed string does not correspond to any name in the enum the last item in the enum will be instantiated and returned.

  Args:
    enum (): The enum class to compare against
    string (str): String representation of the enum member

  Returns:
    enum: Instance of the provided enum class
  """
  for e in enum:
    if e.name == string.upper():
      return e

  print(f'Warning! Unknown type {string}. Assuming {e} data type.')
  return e
