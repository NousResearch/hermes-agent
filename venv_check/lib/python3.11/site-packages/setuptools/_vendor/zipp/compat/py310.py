import sys
import io


def _text_encoding(encoding, stacklevel=2, /):  # pragma: no cover
    return encoding


text_encoding = (
    io.text_encoding if sys.version_info > (3, 10) else _text_encoding  # type: ignore
)
