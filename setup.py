# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
from setuptools import setup
from setuptools import find_packages

setup(
    name='pyem',
    version='0.66',
    packages=find_packages(),
    url='https://github.com/asarnow/pyem',
    license='GNU Public License Version 3',
    author='Daniel Asarnow',
    author_email='asarnow@msg.ucsf.edu',
    description='Python programs for electron microscopy',
    install_requires=['numba', 'numpy', 'numexpr', 'scipy', 'matplotlib',
                      'seaborn', 'pandas', 'pathos', 'pyfftw', 'healpy',
                      'natsort', 'starfile'],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'cfsc.py = pyem.cli.cfsc:_main_',
            'csparc2star.py = pyem.cli.csparc2star:_main_',
            'ctf2star.py = pyem.cli.ctf2star:_main_',
            'disparticle.py = pyem.cli.disparticle:_main_',
            'emcalc.py = pyem.cli.emcalc:_main_',
            'map.py = pyem.cli.map:_main_',
            'mask.py = pyem.cli.mask:_main_',
            'mcstar.py = pyem.cli.mcstar:_main_',
            'par2star.py = pyem.cli.par2star:_main_',
            'project.py = pyem.cli.project:_main_',
            'projection_subtraction.py = pyem.cli.projection_subtraction:_main_',
            'stack.py = pyem.cli.stack:_main_',
            'star.py = pyem.cli.star:_main_',
            'star2bild.py = pyem.cli.star2bild:_main_',
            'subparticles.py = pyem.cli.subparticles:_main_',
            'varmap.py = pyem.cli.varmap:_main_',
        ]
        }
)
