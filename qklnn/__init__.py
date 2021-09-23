# This file is part of qlknn.
# You should have received qlknn LICENSE file with this project.
import pkg_resources

# First thing for import, try to determine imaspy version
try:
    __version__ = pkg_resources.get_distribution("qlknn").version
except Exception:  # pylint: disable=broad-except
    # Try local wrongly install copy
    try:
        from version import __version__
    except Exception:  # pylint: disable=broad-except
        # Local copy or not installed with setuptools.
        # Disable minimum version checks on downstream libraries.
        __version__ = "0.0.0"

import qlknn.setup_logging
