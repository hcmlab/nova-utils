"""Utility module to parse and process strings
Author:
    Dominik Schiller <dominik.schiller@uni-a.de>
Date:
    14.9.2023

"""
from enum import Enum
from typing import Union
from typing import Type
from nova_utils.data.data import Data

def parse_time_string_to_ms(frame: Union[str, int, float, None]) -> int:
    """
    Parse a time string or value to milliseconds.

    This function takes a frame as input, which can be specified as a string in milliseconds (e.g., "100ms"),
    as a string in seconds (e.g., "2s"), as a float in seconds, as an integer in milliseconds, or as None.
    It converts the frame to an integer value in milliseconds.

    Args:
        frame (Union[str, int, float, None]): The frame value to parse.

    Returns:
        int: The frame value in milliseconds. 0 if frame is None.

    Raises:
        ValueError: If the input format for frame is invalid.

    """
    if frame is None:
        return 0

    # if frame is specified milliseconds as string
    if str(frame).endswith("ms"):
        try:
            return int(frame[:-2])
        except ValueError:
            raise ValueError(
                "Invalid input format for frame in milliseconds: {}".format(frame)
            )
    # if frame is specified in seconds as string
    elif str(frame).endswith("s"):
        try:
            frame_s = float(frame[:-1])
            return int(frame_s * 1000)
        except ValueError:
            raise ValueError(
                "Invalid input format for frame in seconds: {}".format(frame)
            )
    # if type is float we assume the input will be seconds
    elif isinstance(frame, float) or "." in str(frame):
        try:
            print(
                "WARNING: Automatically inferred type for frame {} is float.".format(
                    frame
                )
            )
            return int(1000 * float(frame))
        except ValueError:
            raise ValueError("Invalid input format for frame: {}".format(frame))

    # if type is int we assume the input will be milliseconds
    elif isinstance(frame, int) or (isinstance(frame, str) and frame.isdigit()):
        try:
            print(
                "WARNING: Automatically inferred type for frame {} is int.".format(
                    frame
                )
            )
            return int(frame)
        except ValueError:
            raise ValueError("Invalid input format for frame: {}".format(frame))
    else:
        raise ValueError("Invalid input format for frame: {}".format(frame))

def string_to_enum(enum: Enum, string: str):
    """
    Convert a string to an enum value.

    This function takes an enum and a string and returns the corresponding enum value. If the string does not match any enum value,
    a ValueError is raised.

    Args:
        enum (Enum): The enum to search in.
        string (str): The string to convert to an enum value.

    Returns:
        Enum: The enum value corresponding to the input string.

    Raises:
        ValueError: If the input string does not match any enum value in the specified enum.

    """
    for e in enum:
        if e.name == string:
            return e
    raise ValueError('{} not part of enumeration  {}'.format(string, enum))

def string_to_bool(string:str) -> bool:
    '''
    Parses a given input string to a boolean value
    Args:
        string (str): Input string

    Returns:
        bool: The boolean value of the string
    '''
    if isinstance(string, str):
        if string in ['True', 'true', '1']:
            return True
        else:
            return False
    elif isinstance(string, bool):
        return string

def parse_nova_option_string(option_string: str) -> dict:
    """
    Converts a server-module option string to dictionary.

    This function takes an option string as send by nova and converts it to dictionary containing the option name as key and the according value as value.

    Args:
        option_string (str): The option string.

    Returns:
        Enum: The enum value corresponding to the input string.
    """
    options = {}
    print('Parsing options')
    if option_string:
        opts = option_string.split(';')
        for option in opts:
            k,v = option.split('=')
            if v in ("True", "False"):
                options[k] = True if v == "True" else False

            elif v == "None":
                options[k] = None
            else:
                try:
                    options[k] = int(v)
                except:
                    try:
                        options[k] = float(v)
                    except:
                         options[k] = v
            print('\t' + k + "=" + v)
    print('...done')
    return options