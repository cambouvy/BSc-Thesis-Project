# makefile for QLKNN-develop library
PYTHON?=python
PIP?=pip
ECHO?=$(shell which echo) -e
PRINTF?=$(shell which printf)

# Variables required for release and bugfix options
PROJECT=QLKNN-develop
PROJECT_ID=10189900
PACKAGE=qlknn

# Other variables
SUBDIRS:=

#####################################################

.PHONY: help clean sdist bdist wheel install_package_jintrac upload_package_to_gitlab install_package_from_gitlab docs realclean

help:
	@$(ECHO) "Recipes:"
	@$(ECHO) "  sdist                       - build package for source distribution"
	@$(ECHO) "  bdist                       - build package for binary distribution"
	@$(ECHO) "  wheel                       - build package for binary wheel distribution"
	@$(ECHO) "  docs                        - build documentation for package"
	@$(ECHO) "  clean                       - remove all build, test, doc, and Python artifacts"
	@$(ECHO) "  realclean                   - clean and remove build distributions "
	@$(ECHO) ""
	@$(ECHO) "Environment variables:"
	@$(ECHO) "  PYTHON                      - Python binary to use for commands [default: "$(shell grep -e PYTHON?\= Makefile | cut -d\= -f2)"]"
	@$(ECHO) "  PIP                         - Pip binary to use for commands [default: "$(shell grep -e PIP?\= Makefile | cut -d\= -f2)"]"
	@$(ECHO) "  JINTRAC_PYTHON_INSTALL_DIR  - JINTRAC install directory [default: "$(shell grep -e JINTRAC_PYTHON_INSTALL_DIR?\= Makefile | cut -d\= -f2)"]"
	@$(ECHO) "  PYTHONTOOLS_EXTRAS          - Extras to install [default: "$(shell grep -e PYTHONTOOLS_EXTRAS?\= Makefile | cut -d\= -f2)"]"

# Build a 'sdist' or 'installable source distribution' with setuptools
# This creates a sdist package installable with pip
# On pip install this will re-compile from source compiled components
sdist:
	$(PYTHON) setup.py sdist

# Build a 'bdist' or 'installable binary distribution' with setuptools
# This creates a bdist package installable with pip
# This should be pip-installable on the system it was compiled on
# Not recommended to use this! Use wheels instead
bdist:
	$(PYTHON) setup.py bdist

# Build a 'wheel' or 'installable universial binary distribution' with setuptools
# This creates a wheel that can be used with pip
wheel:
	$(PYTHON) setup.py bdist_wheel

# Get the current version.
# The name will be generated from git, i.e.
# qlknn-1.1.1.dev31
# See https://pypi.org/project/setuptools-scm/
# Falls back to `QLKNN_VERSION` env variable if setup.py can't figure it out
# Falls back to 0.0.0 if _that_ does not exist
VERSION_STRING=$(shell $(PYTHON) setup.py --version)
WHEEL_NAME:=$(PACKAGE)-$(VERSION_STRING)-py3-none-any.whl
SDIST_NAME:=$(PACKAGE)-$(VERSION_STRING).tar.gz

docs:
	$(MAKE) -C docs html

clean:
	@echo 'Cleaning $(PROJECT)...'
	$(PYTHON) setup.py clean --all
	$(MAKE) -C docs $@

realclean: clean
	@echo 'Real cleaning $(PROJECT)...'
	rm -f dist/*
	$(MAKE) -C docs $@
