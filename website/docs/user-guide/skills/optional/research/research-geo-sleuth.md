---
title: "Geo Sleuth — Photo geolocation with verified evidence, upstream-kept"
sidebar_label: "Geo Sleuth"
description: "Photo geolocation with verified evidence, upstream-kept"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Geo Sleuth

Photo geolocation with verified evidence, upstream-kept.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/geo-sleuth` |
| Path | `optional-skills/research/geo-sleuth` |
| Version | `2.0.0` |
| Author | Oldcircle (yuanbo) |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `geolocation`, `chronolocation`, `osint`, `geoguessr`, `photo`, `exif`, `ocr`, `reverse-image-search`, `openstreetmap`, `overpass`, `street-view`, `satellite`, `dem`, `sun-shadow` |
| Related skills | [`osint-investigation`](../../optional/research/research-osint-investigation.md), [`domain-intel`](../../optional/research/research-domain-intel.md), [`sherlock`](../../optional/security/security-sherlock.md) |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# geo-sleuth (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [Oldcircle/geo-sleuth](https://github.com/Oldcircle/geo-sleuth): the project
> ships a self-contained skill directory (`skills/geo-sleuth/`) with the
> `SKILL.md`, 20 Python scripts, offline lookup tables and the method
> references. `hermes skills install official/research/geo-sleuth` pulls the
> current tree live from that repo (quarantined and scanned like any hub
> install) — this directory holds only the catalog metadata, so the scripts you
> run can never lag the upstream method.

geo-sleuth geolocates (or chronolocates) a photo with reasoning that is tied to
tool output rather than vibes. One `intake.py` command does metadata, edge and
corner zooms, OCR and reverse image search; every candidate place, clue and
piece of evidence is written to a **candidate board** (`board.py`) whose
ranking, exclusions and "next step" are computed by the script, so a candidate
cannot be dropped without a file that disproves it. Perception steps rank
first and leave the human (or model) to judge only the top few: CLIP-ranked
satellite scans, DINOv2 + SIFT street-view matching, DEM skyline rendering,
camera-pose self-checks, sun/shadow math, and OSM Overpass "feature
combination" searches for places that have structure but no name. The result
is coordinates plus an error radius, an evidence image, and a graded
confidence, each line pointing at the command and file that produced it.

## Prerequisites

- Python 3.10+ and [`uv`](https://docs.astral.sh/uv/) on `PATH`. Every script
  carries an inline PEP 723 dependency block, so `uv run scripts/<x>.py`
  resolves its own packages on first use (the first run adds ~30 s of
  dependency install); nothing is installed into the Hermes environment. The
  core (EXIF, OCR via `rapidocr-onnxruntime`, lookup tables, board, sun, DEM,
  Overpass, gazetteer) runs in seconds; `match.py` and `sat_scan.py`
  pull `torch` + `transformers` (several GB, CPU works) and `revimg.py` needs
  Playwright with a Chromium download. Skip those steps when the machine
  cannot afford them — the skill tells you to mark them "not run", never "not
  found".
- Overpass-backed steps (`osm.py`, `gazetteer.py`, `board.py children` /
  `urban`) take one to four minutes per query even for a county-sized bbox:
  run them with a generous timeout (10 min) or in the background, and use
  native-script place names (`臺灣`, `江苏省`) — English names often do not
  resolve against OSM relations.
- Lookup tables are split by scope: PRC plates, landline area codes and the
  administrative tree are local JSON; calling codes, driving side and
  dependent territories are global. Open `intake/ocr.png` before registering
  OCR text as `read` — low-confidence tile-pass hits are frequently texture
  false positives, which is why the report labels them hypotheses.
- The upstream instructions reference `${CLAUDE_SKILL_DIR}` for the skill's
  own directory (the variable name is the upstream scripts' contract). Hermes
  does not set it: run
  `export CLAUDE_SKILL_DIR=<absolute path of the installed skill>` once per
  shell (or substitute the path) before the first command, exactly as the
  upstream `SKILL.md` tells other agents to do.
- The upstream `SKILL.md` and references are written in **Chinese** (the
  description carries the English triggers); the model reads them fine, but
  users browsing the installed files should expect zh-CN prose and CJK
  examples (Chinese plates, area codes, Baidu panoramas) alongside the global
  tables.
- Network, no API keys: OpenStreetMap Overpass mirrors + Nominatim, AWS
  Terrarium elevation tiles, ArcGIS World Imagery and Google map tiles,
  Google Street View's thumbnail/metadata endpoints, and Baidu / Yandex /
  Sogou reverse image search driven through a browser. **The reverse-image
  step uploads the photo itself to those search engines** (`intake.py`
  without `--no-rev`, or `revimg.py`); pass `--no-rev` when the picture must
  not leave the machine. The tile, Street View and reverse-image paths are
  unofficial consumer endpoints, not documented APIs — they can break or
  rate-limit without notice, and the scripts accept `--proxy` / `GEO_PROXY`
  for regions where they are blocked. The install scanner rates the tree
  `caution` (child processes inherit the parent environment, proxy variables
  are set for the ML scripts); nothing reads credentials.

## Trust and consent

Locating where a stranger's photo was taken is the same capability stalkers
want. The upstream method's rule 0 is binding: when the user did not take the
photo themselves, or the frame shows a private residence or a minor, ask what
the result is for before continuing; proceed without asking only when the
purpose is already stated (a puzzle set, a verification task, the user's own
travel photo). Report "unverified" for anything no command actually checked —
the board enforces this for exclusions, you enforce it for the write-up.

## When to prefer it

- "Where was this taken?", "which city is this?", GeoGuessr-style puzzles,
  dating a photo from shadows or sun position, verifying a claimed location
  for a news or OSINT task.
- Not for people or account discovery (`sherlock`, `osint-investigation`),
  domain and infrastructure intelligence (`domain-intel`), or plain image
  description — `vision_analyze` alone answers "what is in this picture".

Full documentation: https://github.com/Oldcircle/geo-sleuth#readme
