---
title: "Omarchy Integration"
sidebar_label: "Omarchy"
sidebar_position: 5
---

# Omarchy

[Omarchy](https://omarchy.org) is an opinionated Arch + Hyprland desktop built around a Quickshell-based shell ("Quattro"). Its bar has a plugin system, and community plugins already exist for Hermes. This page describes what a desktop integration can rely on and what it should avoid.

## What a desktop shell integration typically uses

- `hermes status` and `hermes doctor` for health checks. Both are safe to run periodically; a cold `hermes` process costs seconds of Python startup, so cache results; the interval is an integration choice (30 minutes is a reasonable ceiling for a version string).
- `hermes gateway start | stop | restart | status` for the messaging gateway's systemd user service. This is the supported control path; do not manage the unit with raw `systemctl` writes.
- `hermes pause` / `hermes resume`, the emergency stop. Pause writes a JSON sentinel at `$HERMES_HOME/ESTOP` (profile-aware: a profile gateway reads its own `HERMES_HOME/ESTOP` first, then the fleet root). Third-party integrations can read that sentinel directly to display the paused state and reason without shelling out.
- `hermes kanban boards list --json` returns machine-readable board summaries for task widgets.
- `hermes send` for one-shot deliveries to configured messaging platforms from scripts and desktop actions.

## Reading local state

The SQLite store at `~/.hermes/state.db` (set `HERMES_HOME` per profile) answers most status questions read-only:

- `sessions`: recent conversations with title, source (cli, telegram, discord, ...), model, message counts, timestamps, git workspace, and per-session cost fields.
- `messages`: per-message timestamps and roles, enough for an activity estimate such as "a message arrived recently" and today's prompt counts.
- `session_model_usage`: per-model token totals and estimated cost, already computed by Hermes.

Open the database with `sqlite3.connect("file:...?mode=ro", uri=True)`. Remember that SQLite columns are dynamically typed: coerce values defensively, because a schema change in a fast-moving codebase should degrade a widget, not crash it. The schema is internal and can change between releases; pin your integration's expectations and fail soft.

## What to avoid

- Do not read `auth.json`, `.env`, or credential stores. If you need provider names, `hermes auth list` prints them; retain only names and counts.
- Do not hand-parse `config.yaml` with regexes for anything beyond simple scalars. `hermes config get <path>` prints resolved values and handles nested structures. A `model:` key can be a scalar, a mapping with a `default:` child, or a dict-format custom-endpoint block; guessing produces wrong answers.
- Do not spawn a Hermes CLI call on a fast refresh timer. Each cold call costs seconds of Python startup. Cache aggressively (TTLs of minutes) and read the database for the frequent path.
- Do not write to `$HERMES_HOME` except through documented commands. Third-party state belongs in `$XDG_STATE_HOME`.

## An example: the Omarchy bar plugin ecosystem

[Hermes Deck](https://github.com/mbot11/omarchy-hermes-deck) is an open-source Quattro plugin that implements the read-only database polling, the action dispatcher with validated arguments, the ESTOP notifications, and the special-workspace TUI modal described above. a persistent state daemon polling the database read-only, an action dispatcher wrapping `hermes pause`/`resume`/`gateway restart` with validated arguments, desktop notifications on emergency-stop and gateway transitions, and a drop-down TUI modal on a Hyprland special workspace. Its `docs/SECURITY.md` documents one concrete trust boundary for desktop shells.
