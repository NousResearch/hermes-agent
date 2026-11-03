"""Extract the Ink host clipboard without loading a session runtime."""
import json
import os
import sys
from pathlib import Path

# The Python root (gatewayClient.ts/imageAttachments.ts pass it): the TUI product ships this
# helper in dist/scripts/, where parents[2] is not the root.
def python_root() -> Path:
    return Path(os.environ.get("HERMES_PYTHON_SRC_ROOT") or Path(__file__).resolve().parents[2])


sys.path.insert(0, str(python_root()))

def main(destination: Path) -> bool:
    # Ink runs this on HERMES_PYTHON with PYTHONPATH=<root> only, which on a source install is PM's
    # store Python: without the committed generation, Pillow is missing and non-PNG clipboard
    # images silently depend on ImageMagick being installed. Activate before any Hermes import, so
    # a module that grows a third-party import still loads from the generation.
    from pm.environments import activate_dependencies

    try:
        activate_dependencies(python_root())
    except (RuntimeError, OSError) as exc:
        # No generation / unreadable install state: no image, not a traceback (Ink reads stdout).
        print(f"clipboard helper: dependency environment unavailable: {exc}", file=sys.stderr)
        return False
    from hermes_cli.clipboard import save_clipboard_image

    return bool(save_clipboard_image(destination))


if __name__ == "__main__":
    print(json.dumps(main(Path(sys.argv[1]))))
