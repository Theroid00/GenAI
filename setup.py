#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for Synthetic Data Generator package.
"""

from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="synthetic_data_gen",
    version="1.0.0",
    author="The Roid",
    author_email="user@example.com",
    description="A versatile tool for generating synthetic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theroid/synthetic-data-generator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.0.0",
        "faker>=8.0.0",
        "tqdm>=4.0.0",
    ],
    extras_require={
        "model": ["sdv>=0.14.0"],
    },
    entry_points={
        "console_scripts": [
            "synthetic-data=synthetic_data_gen.main:main",
        ],
    },
)
