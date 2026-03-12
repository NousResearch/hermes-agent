---
sidebar_position: 10
title: "Audio Tools Troubleshooting"
description: "Diagnose and fix common issues with text-to-speech and transcription"
---

# Audio Tools Troubleshooting

This page collects the most common issues users hit when configuring **text-to-speech (TTS)** and **voice transcription (STT)**, and how to fix them quickly.

## Quick Checklist

- **Virtualenv activated?**
  - Always activate `.venv` before running Hermes or `pytest`:
    ```bash
    source .venv/bin/activate
    ```
  - On Windows PowerShell:
    ```powershell
    .venv\Scripts\Activate.ps1
    ```

- **API keys configured?**
  - TTS:
    - ElevenLabs: `ELEVENLABS_API_KEY`
    - OpenAI: `VOICE_TOOLS_OPENAI_KEY`
  - Transcription:
    - OpenAI: `VOICE_TOOLS_OPENAI_KEY`

- **System dependencies installed?**
  - Telegram voice bubbles with Edge TTS require `ffmpeg`.

## TTS: No Audio or Empty Files

If Hermes reports success but the audio file is missing or zero bytes:

- Make sure at least one TTS provider is correctly installed and configured.
- For **Edge TTS**:
  - Install the Python package:
    ```bash
    uv pip install edge-tts
    ```
  - If you are using Telegram and want voice bubbles, install `ffmpeg` so MP3 output can be converted to Opus/OGG.

Hermes logs the final file path and size when TTS succeeds. If you still see `0 bytes`, check the logs for provider-specific errors.

## TTS: Provider-Specific Failures

- **ElevenLabs**
  - Symptom: immediate failure with a configuration or dependency error.
  - Fixes:
    - Set `ELEVENLABS_API_KEY` in `~/.hermes/.env`.
    - Verify `voice_id` and `model_id` under the `tts.elevenlabs` section of `config.yaml`.

- **OpenAI TTS**
  - Symptom: API error when generating speech.
  - Fixes:
    - Set `VOICE_TOOLS_OPENAI_KEY` in `~/.hermes/.env`.
    - Check that your account has access to the configured TTS model.

## Transcription: Requests Failing

Transcription uses OpenAI’s audio API directly.

- Symptom: Hermes returns an error mentioning `VOICE_TOOLS_OPENAI_KEY` or `API error`.
- Fixes:
  - Set `VOICE_TOOLS_OPENAI_KEY` and ensure it matches your OpenAI account.
  - Keep files under the 25MB limit and in one of the supported formats (`.mp3`, `.ogg`, `.wav`, etc.).

## Where to Look in Logs

Hermes captures detailed error logs (including stack traces) for both TTS and transcription when something goes wrong. If basic fixes do not help:

- Check the main Hermes error log file documented in the **Logs & Diagnostics** section.
- Look for entries tagged with the TTS or transcription module names, which include full exception details to help you or maintainers debug further.

