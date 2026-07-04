---
title: YouTube/video summarizer skill proposal
description: Research-backed design for a transcript-first Hermes skill that summarizes individual YouTube videos safely.
---

# YouTube/video summarizer skill proposal

## Status

- **Proposal only.** No package installs, dependency changes, Docker, scraping, downloading, API keys, or routing changes are included.
- **Recommended outcome:** create a dedicated bundled media skill named `youtube-video-summarizer` rather than widening the existing `youtube-content` skill further.
- **Proposed path:** `skills/media/youtube-video-summarizer/SKILL.md` with optional helpers under `skills/media/youtube-video-summarizer/scripts/` in a later implementation packet.
- **Relationship to existing skill:** `skills/media/youtube-content` can remain a general transform skill for transcripts, threads, blogs, and quotes. The new skill should be the privacy/compliance-focused default for structured video summaries from URLs.

## Research findings

| Area | Finding | Design implication |
| --- | --- | --- |
| `youtube-transcript-api` | PyPI/project docs describe a Python API and CLI for retrieving YouTube transcripts/subtitles, including auto-generated captions, language fallback, translation, transcript metadata (`language`, `language_code`, `is_generated`, `is_translatable`), and JSON/VTT/SRT formatters. It uses an undocumented YouTube web-client API and warns it can break if YouTube changes behavior. The docs also warn about `RequestBlocked`/`IpBlocked`, especially on cloud provider IPs. | Useful first transcript probe when installed, especially because it exposes manual-vs-generated metadata. Treat failures as expected, never retry with proxies/cookies unless separately approved. Mark source as `youtube_transcript_api` and note undocumented-API reliability risk. |
| `yt-dlp` | Project docs describe a broad downloader/extractor. Its subtitle options can write manual subtitles (`--write-subs`) and automatic captions (`--write-auto-subs`), list subtitles, choose languages, and skip media with `--skip-download`. | Good second transcript probe if available, but the skill must constrain it to subtitle metadata/caption extraction only. Any command that downloads audio/video, uses cookies, impersonation, proxies, playlists, or bulk URLs is outside the default flow and requires approval. |
| `whisper.cpp` | Official repo describes a local C/C++ Whisper implementation with no runtime dependencies after build, CPU and accelerated backends, quantized models, and CLI transcription of 16-bit WAV files. | Best privacy-preserving fallback ASR route once preinstalled. It still requires audio extraction/download and model files, so it must be behind an explicit approval gate. Prefer local/offline over cloud ASR for private/sensitive inputs. |
| OpenAI Whisper local | OpenAI Whisper repo provides Python CLI and API (`whisper audio.mp3 --model turbo`, `model.transcribe(...)`), but requires Python dependencies, PyTorch, ffmpeg, and model downloads. | Viable local ASR when already installed, but heavier than `whisper.cpp`. Do not install or download models by default. Treat as local-only unless using an API is separately approved for public/non-sensitive data. |
| YouTube Data API captions | Google docs for `captions.list` and `captions.download` require OAuth scopes. `captions.download` requires that the authorized user has permission to edit the video and costs 200 quota units; `captions.list` costs 50 units and does not contain actual caption data. | Not useful for arbitrary public videos without owner OAuth. Do not make YouTube Data API a default dependency. Consider only for user-owned channels after explicit OAuth/key approval. |
| YouTube Terms of Service | YouTube ToS restricts downloading/reproducing content except as expressly authorized, and restricts automated access such as robots/scrapers except public search engines under robots.txt or with prior written permission. | Skill must be compliance-cautious: single URL only, transcript-first, no bulk playlist/channel extraction, no media download by default, and no circumvention (cookies/proxies/age gates/geo bypass) without explicit user/legal basis. |
| Transcript/ASR quality | Manual captions are usually higher quality than auto captions; automatic captions and ASR can omit punctuation, misrecognize names/numbers, lose speaker attribution, and hallucinate or normalize unclear audio. LLM summarization can further overstate claims if not grounded in transcript spans. | Summary must include transcript source and caveats, separate `key claims` from verified facts, preserve timestamps when available, and state when timestamps are inferred or unavailable. |

## Recommended architecture

### Skill vs worker

| Option | Recommendation | Rationale |
| --- | --- | --- |
| Dedicated skill | **Yes — first packet.** | This is mostly procedural: choose transcript source, apply privacy gates, summarize into a template, and write artifacts. A skill changes agent behavior without adding model-tool schema weight. |
| Worker/subagent | **Optional later.** | Useful only for long videos or batch-like chunk summarization where a worker can produce chunk summaries and an independent verifier can compare summary claims to transcript snippets. Not required for a single-video MVP. |
| Core tool | **No.** | A core tool would add prompt/tool footprint. Terminal + file + skill instructions are enough. |
| Plugin/CLI | **Maybe later.** | If this becomes common and needs robust artifact generation, add a `hermes video-summary` CLI/plugin helper gated by available local dependencies. Keep the first implementation skill-only. |

### Transcript-first flow

1. **Input validation**
   - Accept exactly one YouTube video URL or 11-character video ID.
   - Reject playlists, channel URLs, search URLs, and bulk URL lists.
   - Normalize `youtu.be`, `/watch?v=`, `/shorts/`, `/embed/`, and `/live/` to a video ID.

2. **Metadata probe without media download**
   - Prefer available metadata from transcript tooling output.
   - If `yt-dlp` is already installed, allow metadata-only inspection with `--skip-download --dump-json` only for the single video URL.
   - Capture title, channel/uploader, upload date, duration, webpage URL, and available subtitles when available.

3. **Caption/transcript probe order**
   - `youtube-transcript-api` when installed: list transcripts, prefer manual captions over generated captions, prefer user-requested languages, otherwise use video language then English fallback.
   - `yt-dlp` when installed: use `--skip-download --write-subs --write-auto-subs --sub-langs <langs> --sub-format vtt/srt/json3` to retrieve caption files only, never media.
   - If neither is installed or both fail, stop and ask for approval before any install, API key, cookie/proxy, or audio fallback.

4. **Transcript normalization**
   - Preserve raw transcript/caption artifact.
   - Normalize to JSON segments: `start`, `end`, `duration`, `text`, optional `speaker`, `source_track`, `language`, `is_generated`.
   - Produce timestamped Markdown and plain text for summarization.

5. **Grounded summarization**
   - Chunk long transcripts by timestamp boundaries.
   - For each chunk, summarize only from transcript text; preserve claims with timestamps.
   - Final answer must include caveats and avoid claiming facts not present in transcript.

### Fallback ASR flow

Fallback ASR is not part of the default path because it requires audio extraction/download and usually model files.

1. **Stop and request explicit approval** with the exact command class and data movement:
   - whether audio will be downloaded/extracted;
   - destination path and retention policy;
   - local/offline model to use;
   - whether any network API would receive audio/transcript.
2. **Preferred local route when already installed:** `whisper.cpp` with a local model and local audio conversion.
3. **Secondary local route when already installed:** OpenAI Whisper Python package with local model files.
4. **Cloud ASR route:** only for explicitly `public_non_sensitive` input and separate approval. Never use free/trial/public auxiliary routes for private or sensitive data.
5. **Post-ASR caveat:** label transcript source as `local_asr`, include model name if available, and warn that ASR timestamps/text may be approximate.

## Approval gates

| Gate | Requires separate approval? | Notes |
| --- | --- | --- |
| Public web research about tooling | No | Allowed for design/research. |
| Read-only local skill inspection | No | Allowed. |
| Single-video metadata probe with already-installed tools and no media download | No, if user provided URL for summarization | Must be single URL; no cookies/proxies/playlists. |
| Fetch available captions/transcripts for one provided public URL | No, if tooling is already installed and no circumvention is used | Record source and caveats. |
| Package install or dependency/lockfile change | **Yes** | Never auto-install inside the summarizer skill. |
| Audio/video download or extraction | **Yes** | Even if public; disclose path and retention. |
| Local ASR with existing model/tool | **Yes** | Because it requires audio extraction and may be costly/slow. |
| Cloud ASR or cloud summarization of transcript/audio | **Yes** | Only for `public_non_sensitive`; never for private/sensitive data on public/free/trial routes. |
| Cookies, OAuth, API keys, proxies, age/geo bypass | **Yes, hard gate** | Usually avoid; document legal/compliance risk. |
| Playlist/channel/bulk extraction | **Out of scope for MVP** | Future packet only after separate design review. |

## Privacy and routing rules

- Default to the current private-safe model route already selected by Hermes.
- Treat URLs supplied in private chats as potentially sensitive even if the video is public.
- Do not send transcript/audio/video to public/free/trial auxiliary models unless the user labels the input `public_non_sensitive` and separately approves that route.
- Never print, copy, or store API keys/cookies. Do not use browser cookies by default.
- Local artifacts stay under the Hermes home directory, not inside a project repo, unless the user explicitly requests a repo artifact.

## Local artifact layout

Default base directory:

```text
${HERMES_HOME:-~/.hermes}/artifacts/youtube-video-summaries/<video_id>/<YYYYMMDD-HHMMSS>/
```

Recommended files:

```text
metadata.json              # source URL, title/channel/date/duration, tool versions if known
transcript.raw.<ext>       # original caption/transcript file when available
transcript.normalized.json # canonical segments and source metadata
transcript.timestamped.md  # readable timestamped transcript
summary.md                 # user-facing structured summary
claims.json                # key claims with supporting timestamps and confidence/caveat fields
run.json                   # commands/classes run, approvals, routing, timings, errors
```

Use stable, non-secret names. Do not include video titles in paths unless sanitized.

## Metadata schema

```json
{
  "schema_version": "youtube_video_summary.v1",
  "video": {
    "id": "string",
    "url": "string",
    "title": "string|null",
    "channel": "string|null",
    "channel_url": "string|null",
    "upload_date": "YYYY-MM-DD|null",
    "duration_seconds": 0,
    "language_hint": "string|null"
  },
  "transcript": {
    "source": "youtube_transcript_api|yt_dlp_subtitles|local_asr|user_provided|unavailable",
    "language": "string|null",
    "is_generated": true,
    "is_translated": false,
    "format": "json|vtt|srt|plain|null",
    "segments_count": 0,
    "quality_caveats": ["string"]
  },
  "routing": {
    "summarization_route": "private_safe_default|approved_public_non_sensitive|local_only",
    "asr_route": "none|local_whisper_cpp|local_openai_whisper|approved_cloud_asr",
    "approvals": ["string"]
  },
  "artifacts": {
    "base_dir": "string",
    "metadata": "metadata.json",
    "normalized_transcript": "transcript.normalized.json",
    "summary": "summary.md"
  }
}
```

## Summary template

```markdown
# Video summary: <title or video ID>

| Field | Value |
| --- | --- |
| URL | <url> |
| Channel | <channel or unavailable> |
| Date | <upload date or unavailable> |
| Duration | <duration or unavailable> |
| Transcript source | <manual captions / auto captions / yt-dlp subtitles / local ASR / user-provided> |
| Caveats | <caption/ASR quality, missing metadata, translated transcript, timestamp limits> |

## Concise summary
<5-10 bullets or short paragraphs grounded in the transcript.>

## Timestamped sections
| Time | Section | Notes |
| --- | --- | --- |
| 00:00 | <section> | <what happens> |

## Key claims
| Claim | Evidence timestamp(s) | Caveat |
| --- | --- | --- |
| <claim as stated by speaker> | <mm:ss> | <unverified / ASR uncertain / transcript ambiguous> |

## Action items / useful takeaways
- <practical takeaway>

## Caveats
- <manual/generated/ASR quality note>
- <any missing metadata or inaccessible captions>

## Optional follow-up questions
- <question that would help tailor the output>

## Local artifacts
- `<path/to/summary.md>`
- `<path/to/transcript.normalized.json>`
```

## ToS and compliance notes

- YouTube's Terms restrict downloading/reproducing content except as authorized and restrict automated access such as scrapers except for public search engines under robots.txt or with prior written permission.
- The MVP should avoid media download, playlists, channels, bulk extraction, cookies, proxies, login flows, age/geo bypass, or content-owner APIs.
- `youtube-transcript-api` uses an undocumented YouTube web-client endpoint. This is convenient but fragile and may be blocked; the skill must surface that caveat rather than hiding it.
- YouTube Data API captions are not a general public-caption solution because caption listing/downloading requires OAuth and caption download requires permission to edit the video.
- This proposal is not legal advice; users remain responsible for copyright, platform terms, and permitted use.

## First implementation packet

1. Add `skills/media/youtube-video-summarizer/SKILL.md` with:
   - trigger conditions and single-URL scope;
   - transcript-first tool order;
   - approval gates;
   - summary template;
   - artifact schema/layout;
   - privacy/data-routing rules.
2. Add a helper script only if dependencies are optional and no lockfile changes are needed:
   - normalize video IDs;
   - read transcript output from `youtube-transcript-api` or existing subtitle files;
   - write artifacts under `${HERMES_HOME}/artifacts/youtube-video-summaries/`.
3. Update generated skill docs if the repository requires generated docs in PRs.
4. Add tests for pure helpers only: URL normalization, timestamp formatting, metadata schema validation, and refusal of playlist/channel URLs.
5. Do **not** add installs, Docker, downloader dependencies, routing config, API keys, or actual media download in this packet.

## Verifier checklist

- [ ] Diff is docs/skill-only unless the implementation packet explicitly adds pure helper tests.
- [ ] No `package-lock.json`, dependency, Docker, MCP, plugin, or routing config changes.
- [ ] Skill refuses playlists/channels/bulk URLs by default.
- [ ] Transcript-first order is documented and ASR is behind an explicit approval gate.
- [ ] Audio/video download, cookies, proxies, OAuth, API keys, and cloud ASR/summarization are separate approval gates.
- [ ] Public/free/trial model routes are forbidden for private/sensitive transcript/audio/video data.
- [ ] Summary template includes metadata, transcript source, concise summary, key claims, timestamps, action items/takeaways, caveats, follow-up questions, and local artifact paths.
- [ ] Claims are framed as speaker/video claims unless externally verified.
- [ ] ToS/compliance caveats are present and not overconfident.
- [ ] Local artifacts are stored outside repos by default under Hermes home.

## Recommendation

Create **both** eventually, but sequence them:

1. **Now:** a dedicated Hermes skill, `youtube-video-summarizer`, for safe single-video summaries from URLs.
2. **Later:** an optional worker/helper for long-video chunking, transcript-source comparison, and verifier review if demand justifies it.

Do not make this a core tool. Keep the model-tool footprint unchanged and let the skill orchestrate terminal/file/web capabilities that already exist.
