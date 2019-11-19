"""
setup.py - This module allows for package installation.
"""

from distutils.core import setup

NAME = "rbqoc"
VERSION = "0.1alpha"
DEPENDENCIES = [
    "filelock",
    "h5py",
    "matplotlib",
    "numpy",
]
DESCRIPTION = "This package has utilities for performing robust qoc experiments."
AUTHOR = "Thomas Propson"
AUTHOR_EMAIL = "tcpropson@uchicago.edu"

setup(author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      install_requires=DEPENDENCIES,
      name=NAME,
      version=VERSION,
)
