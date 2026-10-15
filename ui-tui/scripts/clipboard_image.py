"""Extract the Ink host clipboard without loading a session runtime."""
import json
import sys
from pathlib import Path

from hermes_cli.clipboard import save_clipboard_image
from pm.environments import activate_dependencies

if __name__ == "__main__":
    # Ink runs this on HERMES_PYTHON with PYTHONPATH=<root> only, which on a source install is PM's
    # store Python: without the committed generation, Pillow is missing and non-PNG clipboard
    # images silently depend on ImageMagick being installed.
    activate_dependencies(Path(__file__).resolve().parents[2])
    print(json.dumps(bool(save_clipboard_image(Path(sys.argv[1])))))
