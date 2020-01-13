#!/usr/bin/env python
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(packages=[
    'shape_reconstruction',
    'shape_reconstruction.utils',
    'shape_reconstruction.shape_completion',
    'shape_reconstruction.shape_reconstruction_analysis',
],
                                      package_dir={'': 'src'})

setup(**setup_args)
