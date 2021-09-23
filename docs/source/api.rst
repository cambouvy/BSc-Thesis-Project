.. currentmodule:: qlknn

API reference
=============

This page provides an auto-generated summary of QLKNN's API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.

.. See also: :ref:`public api`

QLKNN models
------------

Input
*****

.. autosummary::
   :toctree: generated/
   :template: custom-module-template.rst
   :recursive:

   qlknn.models

Datasets (basic)
**************

We use :std:term:`xarray's Dataset abstraction <xarray:Dataset>`, see :py:class:`xarray:xarray.Dataset`. Prefer using :py:class:`netCDF4 backend <xarray:xarray.backends.NetCDF4DataStore>`, which uses the :netcdf4:`netCDF4 library <>`. Internally, this uses the :netcdf4:`nc4.Dataset <#netCDF4.Dataset>` class and the :netcdf4:`nc4.Dataset.createVariable <#netCDF4.Dataset.createVariable>` method. Test :py:meth:`xarray:xarray.Dataset.to_netcdf`
