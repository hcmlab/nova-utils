"""Data Storage Class for session specific data
Author: Dominik Schiller <dominik.schiller@uni-a.de>
Date: 22.8.2023
"""
import datetime

class Session:
    """
    Class to stores all information belonging to a specific session during processing

    Attributes:
        data (dict, optional): Additional data associated with the session.
        dataset (str, optional): The dataset or category the session belongs to.
        name (str, optional): The name or title of the session.
        duration (int, optional): The duration of the session in minutes.
        location (str, optional): The location or venue of the session.
        language (str, optional): The language used in the session.
        date (datetime, optional): The date and time of the session.
        is_valid (bool, optional): Whether the session is considered valid.

    Args:
        data (dict, optional): Additional data associated with the session.
        dataset (str, optional): The dataset or category the session belongs to.
        name (str, optional): The name or title of the session.
        duration (int, optional): The duration of the session in milliseconds.
        location (str, optional): The location or venue of the session.
        language (str, optional): The language used in the session.
        date (datetime, optional): The date and time of the session.
        is_valid (bool, optional): Whether the session is considered valid.
    """
    def __init__(self, data: dict = None, dataset: str = None, name: str = None, duration: int = None, location: str = None, language: str = None, date: datetime = None, is_valid: bool = True):
        self.data = data
        self.dataset = dataset
        self.name = name
        self.duration = duration
        self.location = location
        self.language = language
        self.date = date
        self.is_valid = is_valid