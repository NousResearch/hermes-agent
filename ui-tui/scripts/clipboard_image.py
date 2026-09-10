"""Extract the Ink host clipboard without loading a session runtime."""
import json
import sys
from pathlib import Path

from hermes_cli.clipboard import save_clipboard_image

if __name__ == "__main__":
    print(json.dumps(bool(save_clipboard_image(Path(sys.argv[1])))))
