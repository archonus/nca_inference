""" Various utility functions for NCA inference"""

import pickle
from typing import Any, Type

import tensorflow as tf


def raise_if(condition: bool, msg: str, exn_type: Type[Exception] = ValueError):
    """
    Raise the exception with the message if the condition is true

    Args:
        condition (bool): Condition to check
        msg (str): Message to use if exception raised
        exn_type (Type[Exception], optional): Exception type to raise. Defaults to ValueError.

    Raises:
        `exn_type`: If the condition is true
    """
    if condition:
        raise exn_type(msg)


def pickle_save(obj: object, file_name: str):
    """
    Save Python object using `pickle` module

    Args:
        obj (object): Object to save
        file_name (str): File name to save to
    """
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(file_name: str) -> Any:
    """
    Load pickled object from file.

    Args:
        file_name (str): Pickle file
    """
    with tf.device("/cpu:0"):  # Ensure that any tensors loaded are loaded on CPU
        with open(file_name, "rb") as f:
            return pickle.load(f)
