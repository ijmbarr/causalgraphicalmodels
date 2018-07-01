"""Setup.py for CausalGraphicalModels"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, 'VERSION')) as version_file:
    version = version_file.read().strip()

setup(
    name="causalgraphicalmodels",
    version=version,
    description="Causality Graphical Models in Python",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/ijmbarr/causalgraphicalmodels",
    author="Iain Barr",
    author_email="iain@degeneratestate.org",
    license="MIT",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.5",
    ],
    keywords="causal inference causal graphical models causality",
    packages=find_packages(exclude=["notebook", "test"]),
    install_requires=["graphviz", "networkx", "numpy", "pandas"]
)
