# This file is part of DUNEdn by M. Rossi
from setuptools import setup, find_packages
from sys import version_info
import os
import re

requirements = ["numpy", "pyyaml", "torch", "torchvision", "matplotlib", "hyperopt"]

PACKAGE = "dunedn"


def get_version():
    """Gets the version from the package's __init__ file
    if there is some problem, let it happily fail"""
    VERSIONFILE = os.path.join("src", PACKAGE, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="dunedn",
    version=get_version(),
    description="ProtoDUNE raw data denoising with DL",
    author="M. Rossi",
    author_email="marco.rossi@cern.ch",
    url="https://github.com/marcorossi5/DUNEdn.git",
    download_url="https://github.com/marcorossi5/DUNEdn/archive/refs/tags/1.0.0.tar.gz",
    entry_points={"console_scripts": ["dunedn = dunedn.scripts.dunedn:main"]},
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    install_requires=requirements,
    classifiers=[
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.6",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
