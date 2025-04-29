"""
Utility functions for handling file operations, including reading/writing YAML and JSON files,
handling binary files and creating directories if they do not exist.
"""

import os
import json
from pathlib import Path
from typing import Any
import base64
import yaml
import joblib
from box import ConfigBox
from ensure import ensure_annotations
# `ensure_annotations` is a library that helps ensure that function annotations
# are used correctly and consistently in Python code.
# For example:
# >>>   def get_product(x: int, y: int):
# >>>       return x * y
# >>>   get_product(2, 3) # Result: 6
# >>>   get_product('2', 3) # Result: '6'
# It provides a decorator that checks the types of function arguments and
# return values at runtime, raising an error if they do not match the specified annotations.

from Chest_Cancer_Classification import logger

@ensure_annotations
def read_yaml(path_to_yaml: str) -> ConfigBox:
    """
    Reads a YAML file and returns its content as a ConfigBox object.

    Args:
        path_to_yaml (str): Path to the YAML file.

    Returns:
        ConfigBox: Content of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml, encoding='utf-8') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info("YAML file %s loaded successfully.", path_to_yaml)
            return ConfigBox(content)
    except (yaml.YAMLError, FileNotFoundError, OSError) as e:
        logger.error("Error reading YAML file %s: %s", path_to_yaml, e)


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Creates directories if they do not exist.

    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool): If True, prints the status of directory creation.
    """
    for directory in path_to_directories:
        os.makedirs(directory, exist_ok=True)
        if verbose:
            logger.info("Directory %s created successfully.", directory)


class JSONHandler:
    """
    A class to handle JSON file operations.
    """
    def __init__(self, path: str, data: dict):
        self.path = path
        self.data = data

    @ensure_annotations
    def save_json(self):
        """
        Saves a dictionary as a JSON file.

        Args:
            path (str): Path to save the JSON file.
            data (dict): Dictionary to save.
        """
        with open(self.path, 'w', encoding='utf-8') as json_file:
            json.dump(self.data, json_file, indent=4)
            logger.info("JSON file %s saved successfully.", self.path)

    @ensure_annotations
    def load_json(self) -> dict:
        """
        Loads a JSON file and returns its content as a dictionary.

        Args:
            path (str): Path to the JSON file.

        Returns:
            dict: Content of the JSON file.
        """
        with open(self.path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            logger.info("JSON file %s loaded successfully.", self.path)
            return data


class BinaryHandler:
    """
    A class to handle binary file operations.
    """
    def __init__(self, path: Path):
        self.path = path

    @ensure_annotations
    def save_bin(self, data: Any):
        """
        Saves data as a binary file.

        Args:
            data (Any): Data to be saved as binary.
        """
        joblib.dump(value=data, filename=self.path)
        logger.info("Binary file saved at: %s", self.path)

    @ensure_annotations
    def load_bin(self) -> Any:
        """
        Loads data from a binary file.

        Returns:
            Any: Object stored in the file.
        """
        data = joblib.load(self.path)
        logger.info("Binary file loaded from: %s", self.path)
        return data


@ensure_annotations
def get_size(file_path: str, unit: str) -> str:
    """
    Returns the size of a file in a human-readable format.

    Args:
        file_path (str): Path to the file.
        unit (str): Unit for size ('KB', 'MB', 'GB').

    Returns:
        str: Size of the file in a human-readable format.
    """
    size = os.path.getsize(file_path)
    if unit == 'KB':
        size = round(size / 1024, 2)
    elif unit == 'MB':
        size = round(size / (1024 ** 2), 2)
    elif unit == 'GB':
        size = round(size / (1024 ** 3), 2)
    else:
        raise ValueError("Invalid unit. Use 'KB', 'MB', or 'GB'.")
    return f"{size:.2f} {unit}"


class ImageBase64Handler:
    """
    A class to handle encoding and decoding of images using Base64.
    """
    def __init__(self):
        pass

    @staticmethod
    @ensure_annotations
    def encode_image_into_base64(image_path: str) -> str:
        """
        Encodes an image file into a Base64 string.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded string of the image.
        """
        with open(image_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
        logger.info("Image at %s encoded into Base64", image_path)
        return encoded_string

    @staticmethod
    @ensure_annotations
    def decode_image(imgstring: str, file_name: str):
        """
        Decodes a Base64 encoded image string and saves it to a file.

        Args:
            imgstring (str): Base64 encoded image string.
            file_name (str): Path to save the decoded image.
        """
        imgdata = base64.b64decode(imgstring)
        with open(file_name, 'wb') as f:
            f.write(imgdata)
        logger.info("Image decoded and saved to %s", file_name)
