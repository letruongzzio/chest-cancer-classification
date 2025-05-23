"""
Python file to configure the project as an installable package, containing information about 
the project name, version, and dependencies.

Instructions:
1. Use `--user` to Install Locally:
    ```bash
    python3 setup.py install --user
    ```
2. Use a Virtual Environment:
    i. Create a virtual environment:
    ```bash
    python3 -m venv env
    ```
    ii. Activate the virtual environment:
    - On Windows:
    ```bash
    .\env\Scripts\activate
    ```
    - On macOS/Linux:
    ```bash
    source env/bin/activate
    ```
    iii. Install the package:
    ```bash
    python3 setup.py install
    ```
    iv. Deactivate the Virtual Environment (when done):
    ```bash
    deactivate
    ```

Recommendation:
Using a virtual environment is the best approach as it avoids permission issues and
keeps your project dependencies isolated.
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "chest-cancer-classification"
AUTHOR_USER_NAME = "letruongzzio"
SRC_REPO = "Chest-Cancer-Classification"
AUTHOR_EMAIL = "lephutruong.2210@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for chest cancer classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
