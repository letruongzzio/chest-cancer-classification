"""
This script creates a directory structure and files for a machine learning project.
It sets up a directory structure with necessary files for a machine learning project.
It includes directories for components, utils, config, pipeline, entity, constants.
"""

import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s:')

PROJECT_NAME = "Chest_Cancer_Classification"

list_of_files = [
    # Main role: Placeholder for the workflows directory of GitHub Actions to ensure this directory is committed to Git, even if empty. This directory typically contains YAML files defining CI/CD workflows.
    ".github/workflows/.gitkeep",
    
    # Main role: Python initialization file to mark the src/{project_name} directory as a Python package, allowing modules in the project to be imported.
    f"src/{PROJECT_NAME}/__init__.py",
    
    # Main role: Python initialization file for the components directory, marking it as a subpackage containing modules related to components such as data ingestion, model training, and evaluation.
    f"src/{PROJECT_NAME}/components/__init__.py",
    
    # Main role: Python initialization file for the utils directory, marking it as a subpackage containing shared utility functions such as reading YAML, saving JSON, or data processing functions.
    f"src/{PROJECT_NAME}/utils/__init__.py",
    
    # Main role: Python initialization file for the config directory, marking it as a subpackage containing modules related to project configuration.
    f"src/{PROJECT_NAME}/config/__init__.py",
    
    # Main role: Contains classes or functions to manage project configuration, read parameters from the config.yaml file, and provide configuration to other components (data ingestion, training, evaluation).
    f"src/{PROJECT_NAME}/config/configuration.py",
    
    # Main role: Python initialization file for the pipeline directory, marking it as a subpackage containing modules defining training and prediction pipelines.
    f"src/{PROJECT_NAME}/pipeline/__init__.py",
    
    # Main role: Python initialization file for the entity directory, marking it as a subpackage containing data classes such as configurations for pipeline steps.
    f"src/{PROJECT_NAME}/entity/__init__.py",
    
    # Main role: Python initialization file for the constants directory, marking it as a subpackage containing constants such as file paths or MLflow URIs.
    f"src/{PROJECT_NAME}/constants/__init__.py",
    
    # Main role: YAML file containing the main configuration parameters of the project, such as data paths, image sizes, batch size, or MLflow URI.
    "config/config.yaml",
    
    # Main role: YAML file defining the DVC pipeline, including stages such as data ingestion, training, evaluation, along with commands, dependencies, and outputs.
    "dvc.yaml",
    
    # Main role: YAML file containing model training parameters (e.g., learning rate, epochs, batch size) for easy adjustment and tracking via DVC.
    "params.yaml",
    
    # Main role: Python file to configure the project as an installable package, containing information about the project name, version, and dependencies.
    "setup.py",
    
    # Main role: Jupyter Notebook containing initial experiments or data analysis, typically used during the research phase before transitioning to formal code.
    "research/trials.ipynb",
    
    # Main role: Sample HTML file for the web interface of the application (if any), typically used with Flask to display a user interface for predictions.
    "templates/index.html"
]

for file_path in list_of_files:
    file_dir = os.path.dirname(file_path)

    # Create directories if they do not exist and file_dir is not empty
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir)
        logging.info("Directory created: %s", file_dir)
    elif file_dir:
        logging.info("Directory already exists: %s", file_dir)
    
    # Create files if they do not exist
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass
        logging.info("File created: %s", file_path)
    else:
        logging.info("File already exists: %s", file_path)

