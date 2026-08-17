"""Regression: blind-detector must catch the real Gemini refusal payloads
that slipped past _VIDEO_BLIND_RE on 2026-08-17 (me group, vid_6b992e19f3e3).

Both refusals returned success=True method=video_url and were relayed to the
parent as if they were analyses. Payloads below are verbatim from session
20260817_122931_50618817 (msgs 486925 / 486927).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.vision_tools import _video_analysis_is_blind

# 2202-char refusal (call 1, 04:22:53) — abbreviated to the load-bearing prose;
# structure preserved: apology + capability denial + hypothetical checklist.
BLIND_1 = (
    "I apologize, but as an AI, I do not have the ability to view or process "
    "video attachments directly. My input is text-based, and I cannot \"look "
    "at\" or analyze the bytes of a video file. Therefore, I cannot describe "
    "what is visibly present in the video you are referring to.\n\n"
    "If you can provide a detailed textual description of the video's content, "
    "including all visual elements, motion, audio cues, and on-screen text, I "
    "would be happy to help you analyze it and answer your specific questions "
    "about it being an Instagram story/repost.\n\n"
    "To answer your question about the Instagram story/repost, if I were able "
    "to view the video, I would specifically look for and transcribe the "
    "following:\n\n"
    "*   **Account Name:** I would identify and transcribe the Instagram "
    "account name (e.g., \"@username\") that posted or reposted the story.\n"
    "Please provide a textual description of the video if you would like me to "
    "analyze it further."
)

# 605-char refusal (call 2, 04:23:02) — verbatim.
BLIND_2 = (
    "I apologize, but as an AI, I do not have the ability to \"look at an "
    "actual video attachment (bytes attached)\" or process visual information "
    "from video files. My capabilities are limited to processing and "
    "generating text.\n\n"
    "Therefore, I cannot describe what is visibly present in the video, quote "
    "on-screen text, describe frame by frame, or transcribe any content from "
    "it. I cannot fulfill any part of your request that requires analyzing a "
    "video file.\n\n"
    "If you can provide a text-based description of the video, including any "
    "on-screen text, I would be happy to help you analyze or discuss that "
    "information."
)

# Real analyses must NOT be flagged (guard against false positives).
GROUNDED_1 = (
    "The video is a screen recording of an Instagram story. The account "
    "@gina.landycoo.bha3193 posted a 7-slide carousel titled 美人无相. Slide 1 "
    "shows white serif text on a beige background reading 向内觉察，守好本心. "
    "A song overlay shows 半岛铁盒 - 周杰伦. The recording lasts about 12 "
    "seconds and ends on slide 4/7."
)
GROUNDED_2 = (
    "This clip shows a phone UI screen recording. On-screen text (verbatim): "
    "\"I picked you\". The user cannot be seen; only the feed is visible. "
    "Note: the video has no audio track, so no audio cues can be reported."
)


def test_real_refusal_1_detected():
    assert _video_analysis_is_blind(BLIND_1), "2202-char refusal slipped past detector"


def test_real_refusal_2_detected():
    assert _video_analysis_is_blind(BLIND_2), "605-char refusal slipped past detector"


def test_grounded_analysis_not_flagged():
    assert not _video_analysis_is_blind(GROUNDED_1)


def test_grounded_analysis_with_negations_not_flagged():
    # Contains "cannot" and "no audio" in a legitimate reading — must not flag.
    assert not _video_analysis_is_blind(GROUNDED_2)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
