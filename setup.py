# Copyright (C) 2016 Daniel Asarnow
# University of California, San Francisco
from setuptools import setup
from setuptools import find_packages

setup(
    name='pyem',
    version='0.4',
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
            'pyem_csparc2star.py = scripts.csparc2star:_main_',
            'pyem_star.py = scripts.star:_main_',
            'pyem_recenter.py = scripts.recenter:_main_',
            'pyem_angdist.py = scripts.angdist:_main_',
            'pyem_cfsc.py = scripts.cfsc:_main_',
            'pyem_ctf2star.py = scripts.ctf2star:_main_',
            'pyem_emcalc.py = scripts.emcalc:_main_',
            'pyem_map.py = scripts.map:_main_',
            'pyem_mask.py = scripts.mask:_main_',
            'pyem_par2star.py = scripts.par2star:_main_',
            'pyem_pose.py = scripts.pose:_main_',
            'pyem_project.py = scripts.project:_main_',
            'pyem_projection_subtraction.py = scripts.projection_subtraction:_main_',
            'pyem_reconstruct.py = scripts.reconstruct:_main_',
            'pyem_sort.py = scripts.sort:_main_',

        ]
        }
)
