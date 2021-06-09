import os
import sys
from distutils.sysconfig import get_python_lib

from distutils.core import setup
from setuptools import find_packages

CURRENT_PYTHON = sys.version_info[:2]

setup(name="pyfmri", version="0.2",
      description="A package for learning to classify raw audio according to a user's self-definied "
                  "scores of appreciation",
      url="https://github.com/pytorch/audio",
      packages=find_packages(),
      author="Simon J Pelletier",
      author_email="simonjpelletier@gmail.com",
      install_requires=['torch',
                        'matplotlib',
                        'numpy',
                        'scipy',
                        'Unidecode',
                        'nibabel',
                        'nilearn',
                        'tqdm',
                        'torchvision',
                        'ax-platform',
                        'tensorboardX'
                        # 'apex'
                        ]
      )
