import os
from box.exceptions import BoxValueError
import yaml
from cnnClassifier import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file {path_to_yaml} loadded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True) -> None:
    for dirs in path_to_directories:
        os.makedirs(dirs, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path_to_directories}")

@ensure_annotations
def save_json(path: Path, data: dict) -> None:
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Json saved at: {path}")

@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    with open(path, 'r') as f:
        content = json.load(f)
    logger.info(f"Successfully loaded json from: {path}")
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

@ensure_annotations
def load_bin(path: Path):
    joblib.load(filename=path)
    logger.info(f"Successfully loaded binary from: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024, 2)
    return f"{size_in_kb} kb"

def decodeImage(imgString, fileName) -> None:
    imgData = base64.b64decode(imgString)
    with open(fileName, 'wb') as f:
        f.write(imgData)
        f.close

def encodeImage(imagePath) -> bytes:
    with open(imagePath, 'rb') as f:
        imgData = f.read()
        return base64.b64encode(imgData)