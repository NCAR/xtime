#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

with open('README.rst', 'r') as fh:
    long_description = fh.read()

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
]

setup(
    name='xtimeutil',
    description='A Python package for managing time axis and different operations related to time with xarray',
    long_description=long_description,
    python_requires='>=3.6',
    maintainer='NCAR XDev Team',
    maintainer_email='xdev@ucar.edu',
    classifiers=CLASSIFIERS,
    url='https://github.com/NCAR/xtimeutil',
    packages=find_packages(exclude=('tests',)),
    include_package_data=True,
    install_requires=install_requires,
    license='Apache-2.0',
    zip_safe=False,
    entry_points={},
    keywords='xarray, cftime, time',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0'],
)
