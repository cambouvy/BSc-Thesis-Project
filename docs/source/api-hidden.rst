.. Generate API reference pages, but don't display these in tables.
.. This extra page is a work around for sphinx not having any support for
.. hiding an autosummary table.

API autosummary
===============

.. Explicitly list submodules here
.. autosummary::
    :toctree: generated/
    :recursive:
    :template: custom-module-template.rst

    qlknn.dataset
    qlknn.gui
    qlknn.misc
    qlknn.models
    qlknn.NNDB
    qlknn.pipeline
    qlknn.plots
    qlknn.training
    qlknn.__init__.py
    qlknn.setup_logging.py
    qlknn.version.py
