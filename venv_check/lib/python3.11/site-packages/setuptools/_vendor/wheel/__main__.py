"""
Wheel command line tool (enable python -m wheel syntax)
"""

from __future__ import annotations

import sys


def main():  # needed for console script
    if __package__ == "":
        # To be able to run 'python wheel-0.9.whl/wheel':
        import os.path

        path = os.path.dirname(os.path.dirname(__file__))
        sys.path[0:0] = [path]
    import wheel.cli

    sys.exit(wheel.cli.main())


if __name__ == "__main__":
    sys.exit(main())
