# This file is part of QLKNN-develop
import sys

if sys.version_info < (3, 6):
    sys.exit(
        "Sorry, Python < 3.6 is not supported. Use a different"
        " python e.g. `module swap python python/3.7`"
    )

# pylint: disable=wrong-import-position
import os
import logging
from pathlib import Path
from itertools import chain
import ast
import site

# Use setuptools to build packages
from setuptools import find_packages, setup

# Import distutils, as advised by setuptools, after setuptools import
import distutils.text_file

# Only use root logger for convenience
package_name = "qlknn"
root_logger = logging.getLogger(package_name)
logger = root_logger
logger.setLevel(logging.INFO)

# Get path to file, we need this for the rest of setup
this_file = Path(__file__)
this_dir = this_file.parent

# Workaround for https://github.com/pypa/pip/issues/7953
# Cannot install into user site directory with editable source
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Use README as long description for display on PyPI
with open(this_dir / "README.md") as f:
    long_description = f.read()

# As long as Python packaging is in flux, use something both pip and Conda
# can understand
optional_reqs = {}
for file in Path(".").glob("requirements_*.txt"):
    req = file.name.replace("requirements_", "").replace(".txt", "")
    optional_reqs[req] = distutils.text_file.TextFile(this_dir / file).readlines()
install_requires = optional_reqs.pop("core")
# collect all optional dependencies in a "all" target
optional_reqs["all"] = list(chain(*optional_reqs.values()))

if __name__ == "__main__":
    # Legacy setuptools support, e.g. `python setup.py something`
    # See [PEP-0517](https://www.python.org/dev/peps/pep-0517/) and
    # [setuptools docs](https://setuptools.readthedocs.io/en/latest/userguide/quickstart.html#basic-use)
    pyproject: list = distutils.text_file.TextFile("pyproject.toml").readlines()
    requires_line: str = [line for line in pyproject if "requires =" in line][0]
    requires: str = requires_line.split("=", 1)[1]
    setup_requires: list = ast.literal_eval(requires.strip())

    setup(
        name=package_name,
        description="Tools to create QuaLiKiz Quasi-linear gyrokinetic code Neural Networks",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://gitlab.com/Karel-van-de-Plassche/QLKNN-develop",
        author="Karel van de Plassche",
        author_email="k.l.vandeplassche@differ.nl",
        classifiers=[
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
            "Natural Language :: English",
            "Topic :: Utilities",
        ],
        packages=find_packages(),
        # Include files specified by MANIFEST.in
        include_package_data=True,
        # No pyproject.toml for --no-build-installation. Use setup.py instead
        use_scm_version={
            "write_to": package_name + "/version.py",
            "write_to_template": '"{version}"',
            "relative_to": this_file,
            # For tarball installs without metadata (e.g. .git repository)
            "version_scheme": "guess-next-dev",
            "local_scheme": "no-local-version",
            "fallback_version": os.getenv("QLKNN_VERSION", "0.0.0"),
        },
        python_requires=">=3.6",
        # Duplicate from pyproject.toml for older setuptools
        setup_requires=setup_requires,
        install_requires=install_requires,
        extras_require=optional_reqs,
        entry_points={
            "console_scripts": [
                "qlknn=qlknn.cli:main",
                "quickslicer=qlknn.plots.quickslicer:main",
                "clustering=qlknn.dataset.clustering:main",
            ]
        },
    )
