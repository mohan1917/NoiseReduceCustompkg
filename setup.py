"""
Minimal shim so `pip install -e .` works even on older setuptools
versions that don't support PEP 660 editable installs via pyproject.toml
alone. All real metadata lives in pyproject.toml.
"""
from setuptools import setup

setup()
