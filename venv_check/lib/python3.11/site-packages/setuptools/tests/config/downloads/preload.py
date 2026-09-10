"""This file can be used to preload files needed for testing.

For example you can use::

    cd setuptools/tests/config
    python -m downloads.preload setupcfg_examples.txt

to make sure the `setup.cfg` examples are downloaded before starting the tests.
"""

import sys
from pathlib import Path

from . import retrieve_file, urls_from_file

if __name__ == "__main__":
    urls = urls_from_file(Path(sys.argv[1]))
    list(map(retrieve_file, urls))
