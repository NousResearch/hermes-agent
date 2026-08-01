"""Tests for the playback-phase TTS-echo guard (#75780).

Contract:
  - `is_tts_echo` flags a barge-in transcript as a likely self-capture of
    Hermes' own TTS output when it is a close character-level match for the
    text that was just spoken, regardless of language/tokenization.
  - A genuine, unrelated user interjection captured during playback must
    NOT be flagged, even though it happens to share some words.
  - `HermesCLI._voice_submit_barge_utterance` uses this guard ONLY for
    playback-phase barge captures (generation-phase speech can't be TTS
    bleed, since nothing is playing) and drops the echoed transcript
    instead of queuing it as the next user turn.
"""

from tools.voice_mode import is_tts_echo


class TestIsTtsEcho:
    def test_near_verbatim_repeat_is_echo(self):
        spoken = (
            "맞아요. 사용자가 마이크를 끄는 게 아니라 앱이 제 음성은 "
            "에코 제거로 걸러내고 실제 사용자 음성만 끼어들기로 받아야 해요."
        )
        transcript = spoken
        assert is_tts_echo(transcript, spoken) is True

    def test_repeat_with_leading_stutter_is_echo(self):
        spoken = "네, 방금도 제 답변이 그대로 다시 입력됐어요."
        transcript = "네 방금 네 방금도 제 답변이 그대로 다시 입력됐어요."
        assert is_tts_echo(transcript, spoken) is True

    def test_unrelated_interjection_is_not_echo(self):
        spoken = "The weather today is sunny with a light breeze from the west."
        transcript = "actually can you also check my calendar for tomorrow"
        assert is_tts_echo(transcript, spoken) is False

    def test_short_unrelated_reply_is_not_echo(self):
        spoken = "I've finished summarizing the document you shared earlier."
        transcript = "stop"
        assert is_tts_echo(transcript, spoken) is False

    def test_short_fragment_of_longer_multi_sentence_reply_is_echo(self):
        # Playback-phase captures are cut immediately on trigger and only
        # span pre-roll + time-to-silence, so a real self-capture is
        # typically a short fragment of a much longer spoken reply, not a
        # near-verbatim repeat of the whole thing. A whole-string ratio
        # dilutes with the length mismatch and misses this case (#75780
        # review).
        spoken = (
            "Sure, here's a summary of what we found. The build failed "
            "because of a missing dependency in the lockfile. I've already "
            "gone ahead and regenerated it, and the tests are passing "
            "again locally. Let me know if you'd like me to open a PR for "
            "this or if you want to review the diff first before I do "
            "anything else."
        )
        transcript = "Sure, here's a summary of what we found."
        assert is_tts_echo(transcript, spoken) is True

    def test_short_fragment_from_middle_of_reply_is_echo(self):
        spoken = (
            "The deployment finished successfully. All three services "
            "came up healthy, and the smoke tests passed without any "
            "errors."
        )
        transcript = "the smoke tests passed without any errors"
        assert is_tts_echo(transcript, spoken) is True

    def test_empty_inputs_are_not_echo(self):
        assert is_tts_echo("", "hello") is False
        assert is_tts_echo("hello", "") is False
        assert is_tts_echo("", "") is False

    def test_case_and_whitespace_insensitive(self):
        spoken = "Sure, I can help with that right away."
        transcript = "  SURE,   I can help with that   right away.  "
        assert is_tts_echo(transcript, spoken) is True

    def test_custom_threshold_is_honored(self):
        spoken = "This is a moderately similar sentence about testing."
        transcript = "This is a rather different sentence about coding."
        # Lenient threshold treats it as an echo, strict threshold does not.
        assert is_tts_echo(transcript, spoken, threshold=0.5) is True
        assert is_tts_echo(transcript, spoken, threshold=0.95) is False
