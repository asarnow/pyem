# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
from distutils.core import setup

setup(
    name='pyem',
    version='0.1',
    packages=['pyem'],
    scripts=['angdist', 'project', 'projection_subtraction', 'recenter', 'pyem/star'],
    url='https://github.com/asarnow/pyem',
    license='GNU Public License Version 3',
    author='Daniel Asarnow',
    author_email='dasarnow@gmail.com',
    description='Python programs for electron microscopy',
    install_requires=['numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas', 'pathos']
)
