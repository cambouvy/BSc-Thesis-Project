#!/bin/sh
# Create a python venv with access to the globally installed packages.
# This links to setuptools, pip and python, pkg_resources, easy_install as well
python -m venv --system-site-packages $PYTHONTOOLS_VENV_PATH
source $PYTHONTOOLS_VENV_PATH/bin/activate
# Simulate Python3.9 --upgrade-deps flag
#   --upgrade-deps        Upgrade core dependencies: pip setuptools to the
#                         latest version in PyPI
python -m pip install --force-reinstall --upgrade pip setuptools wheel

# Give venv PYTONPATH and PATH precedence over pre-defined env ones
export PATH=$VIRTUAL_ENV/bin:$PATH
export PYTHONPATH=$VIRTUAL_ENV/lib/python`python --version | cut -d' ' -f2 | cut -d'.' -f1-2`/site-packages/:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages/ # Add system packages
