"""Streaming speech cuts a reply into sentences before markdown cleanup, so a block that is
never spoken (a code fence, a reasoning block) must be held by the chunker itself."""

import pytest

from tools.tts_streaming import SentenceChunker
from tools.tts_text_normalize import _strip_markdown_for_tts, prepare_spoken_text

_BEFORE = "Here is a script that cleans the build folder for you.\n\n"
_AFTER = "\n\nRun it from the project root and it will delete the folder."


@pytest.mark.parametrize("block", [
    "```python\nimport shutil\n\nshutil.rmtree('build')\nprint('Removed the build folder. All done!')\n```",
    "<think>The user wants the folder gone. I will say so. Then explain the command.</think>",
])
@pytest.mark.parametrize("step", [1, 7])
def test_streamed_reply_speaks_what_the_whole_reply_speaks(block, step):
    """However the deltas cut the block, and even when the producer stalls inside it (an idle
    flush, which is not end-of-text), a streamed reply speaks exactly the whole reply's script."""
    reply = _BEFORE + block + _AFTER
    chunker, sentences, stalled = SentenceChunker(), [], False
    for i in range(0, len(reply), step):
        sentences += chunker.feed(reply[i:i + step])
        if not stalled and i > len(_BEFORE) + 12:
            sentences += chunker.flush()
            stalled = True
    sentences += chunker.flush()

    spoken = " ".join(filter(None, (_strip_markdown_for_tts(s).strip() for s in sentences)))
    assert spoken == prepare_spoken_text(reply)
