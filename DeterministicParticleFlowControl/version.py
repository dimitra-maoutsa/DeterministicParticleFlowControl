from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 3
_version_micro = ''  # use '' for first of series, number for 1 and above
#_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "DeterministicParticleFlowControl: deterministic particle based stochastic optimal control framework"
# Long description will go up on the pypi page
long_description = """
Deterministic Particle Flow Control
====================================

``Deterministic Particle Flow Control`` is a python package that allows for efficient 
and one shot computation of interventions for optimal stochastic control problems.

Details
=======

Contains code for control computations on systems whose dynamics are described by stochastic differential equations.
To get started, please go to the
repository README file: https://github.com/dimitra-maoutsa/DeterministicParticleFlowControl/blob/main/README.md


License
=======


``DeterministicParticleFlowControl`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
Copyright (c) 2021--, Dimitra Maoutsa, Technical University of Berlin.
"""

NAME = "DeterministicParticleFlowControl"
MAINTAINER = "Dimitra Maoutsa"
MAINTAINER_EMAIL = "dimitra.maoutsa@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
LONG_DESCRIPTION = long_description
URL = "https://github.com/dimitra-maoutsa/DeterministicParticleFlowControl"
DOWNLOAD_URL = "https://github.com/dimitra-maoutsa/DeterministicParticleFlowControl/archive/refs/tags/v0.1.1.tar.gz"
LICENSE = "MIT"
AUTHOR = "Dimitra Maoutsa"
AUTHOR_EMAIL = "dimitra.maoutsa@tu-berlin.de"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'DeterministicParticleFlowControl': [pjoin('data', '*')]}
REQUIRES = ["numpy","scipy","matplotlib","numba","POT","pyemd"]    
PYTHON_REQUIRES = ">= 3.5"
