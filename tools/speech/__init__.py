"""Speech infrastructure: provider-agnostic TTS routing.

See the project's "RFC: Speech Router" design document for the full
architecture (SpeechChunker, SpeechRouter, SpeechProvider, SpeechPlayer).
Modules land incrementally, one small PR at a time; this package currently
only contains SpeechRouter (router.py).
"""
