
xgriddedaxis: Cell Boundary-aware Operations with xarray
========================================================

|pypi| |conda forge| |Build Status| |codecov| |docs| |GitHub Workflow Status|


**xgriddedaxis** is a Python package for managing/working with one-dimensional axes with their respective
cell boundaries information. xgriddedaxis consumes and produces xarray_ data structures,
which are coordinate and metadata-rich representations of multidimensional array data.


xgriddedaxis was motivated by the fact that xarray_ is not aware of cell boundary variables when
performing operations such as **resampling**. The main objective of xgriddedaxis is to provide a set of
utilities that enables fluid translation between data at different intervals while being aware
of the cell boundary variables.

The fundamental concept in xgriddedaxis is a `Remapper` object. `Remapper`'s role includes:

- Creating a source axis, i.e. the axis that your original data is on,
- Creating a destination axis, i.e. the axis that you want to convert your data to,
- Creating a `Remapper` object by passing the source and destination axis you created previously,
- Finally, converting your data from the source axis to the destination axis, using the `Remapper` object you created in previous step.

For more information, read the full
`xgriddedaxis documentation`_.

Installation
------------

xgriddedaxis can be installed from PyPI with pip:

.. code-block:: bash

    python -m pip install xgriddedaxis


It is also available from `conda-forge` for conda installations:

.. code-block:: bash

    conda install -c conda-forge xgriddedaxis


.. _xarray: http://xarray.pydata.org
.. _xgriddedaxis documentation: https://xgriddedaxis.readthedocs.io

.. |GitHub Workflow Status| image:: https://img.shields.io/github/workflow/status/NCAR/xgriddedaxis/code-style?label=Code%20Style&style=for-the-badge
    :target: https://github.com/NCAR/xgriddedaxis/actions

.. |Build Status| image:: https://img.shields.io/circleci/project/github/NCAR/xgriddedaxis/master.svg?style=for-the-badge&logo=circleci
    :target: https://circleci.com/gh/NCAR/xgriddedaxis/tree/master

.. |codecov| image:: https://img.shields.io/codecov/c/github/NCAR/xgriddedaxis.svg?style=for-the-badge
    :target: https://codecov.io/gh/NCAR/xgriddedaxis

.. |docs| image:: https://img.shields.io/readthedocs/xgriddedaxis/latest.svg?style=for-the-badge
    :target: https://xgriddedaxis.readthedocs.io/en/latest/?badge=latest

.. |pypi| image:: https://img.shields.io/pypi/v/xgriddedaxis.svg?style=for-the-badge
    :target: https://pypi.org/project/xgriddedaxis

.. |conda forge| image:: https://img.shields.io/conda/vn/conda-forge/xgriddedaxis.svg?style=for-the-badge
    :target: https://anaconda.org/conda-forge/xgriddedaxis
