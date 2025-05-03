"""
This module contains constants used in the project.
It includes file paths for configuration and parameters files.
It is recommended to use relative paths for better portability.
"""

from pathlib import Path

def find_project_root(filename="config/config.yaml") -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / filename).exists():
            return parent
    raise FileNotFoundError(f"Could not find {filename} in any parent directory.")

PROJECT_ROOT = find_project_root()
CONFIG_FILE_PATH = PROJECT_ROOT / "config/config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"


print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"CONFIG_FILE_PATH: {CONFIG_FILE_PATH}")
print(f"PARAMS_FILE_PATH: {PARAMS_FILE_PATH}")
