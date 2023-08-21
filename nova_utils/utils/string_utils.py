from typing import Union


def parse_time_string_to_ms(frame: Union[str, int, float, None]) -> int:

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
