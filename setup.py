# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
from setuptools import setup
from setuptools import find_packages

entrypoint_prefix= "pyem_"
setup(
    name='pyem',
    version='0.6b1',
    packages=find_packages(),
    url='https://github.com/asarnow/pyem',
    license='GNU Public License Version 3',
    author='Daniel Asarnow',
    author_email='asarnow@msg.ucsf.edu',
    description='Python programs for electron microscopy',
    install_requires=['future', 'numba', 'numpy', 'scipy', 'matplotlib',
                      'seaborn', 'pandas', 'pathos', 'pyfftw', 'healpy', 'natsort'],
    zip_safe=False,
    entry_points={
        'console_scripts': [
            f'{entrypoint_prefix}csparc2star.py = scripts.csparc2star:_main_',
            f'{entrypoint_prefix}star.py = scripts.star:_main_',
            f'{entrypoint_prefix}recenter.py = scripts.recenter:_main_',
            f'{entrypoint_prefix}angdist.py = scripts.angdist:_main_',
            f'{entrypoint_prefix}cfsc.py = scripts.cfsc:_main_',
            f'{entrypoint_prefix}ctf2star.py = scripts.ctf2star:_main_',
            f'{entrypoint_prefix}emcalc.py = scripts.emcalc:_main_',
            f'{entrypoint_prefix}map.py = scripts.map:_main_',
            f'{entrypoint_prefix}mask.py = scripts.mask:_main_',
            f'{entrypoint_prefix}par2star.py = scripts.par2star:_main_',
            f'{entrypoint_prefix}pose.py = scripts.pose:_main_',
            f'{entrypoint_prefix}project.py = scripts.project:_main_',
            f'{entrypoint_prefix}projection_subtraction.py = scripts.projection_subtraction:_main_',
            f'{entrypoint_prefix}reconstruct.py = scripts.reconstruct:_main_',
            f'{entrypoint_prefix}sort.py = scripts.sort:_main_',
            f'{entrypoint_prefix}stack.py = scripts.stack:_main_',
            f'{entrypoint_prefix}star2bild.py = scripts.star2bild:_main_',
            f'{entrypoint_prefix}subparticles.py = scripts.subparticles:_main_',
            f'{entrypoint_prefix}subset.py = scripts.subset:_main_',
            f'{entrypoint_prefix}varmap.py = scripts.varmap:_main_',
        ]
        }
)
