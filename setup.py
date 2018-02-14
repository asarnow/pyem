# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
from setuptools import setup
from setuptools import find_packages

setup(
    name='pyem',
    version='0.1',
    packages=find_packages(),
#    scripts=['angdist', 'project', 'projection_subtraction', 'recenter', 'pyem/star'],
#    scripts=['pyem/star'],
    url='https://github.com/asarnow/pyem',
    license='GNU Public License Version 3',
    author='Daniel Asarnow',
    author_email='dasarnow@gmail.com',
    description='Python programs for electron microscopy',
    install_requires=['future', 'numba', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas', 'pathos']
)
