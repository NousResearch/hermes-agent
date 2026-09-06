---
title: Realms (optional plugin)
description: Separate Linux GUI desktops for parallel Hermes conversations, disabled by default.
---

# Realms

Realms gives each conversation its own Linux desktop so parallel agents can work without moving your windows or competing for your pointer and focus. It is especially useful with tiling desktops such as Hyprland. **Realms is not a VM or security sandbox**; ordinary realm terminals retain development filesystem and network access.

:::caution Prerequisite proposal
This bundled-plugin proposal depends on [PR #103690](https://github.com/NousResearch/hermes-agent/pull/103690) merging first. It must not be presented as compatible with releases without the generic session-extension APIs.
:::

Realms ships as one native plugin package, **disabled by default**. Discovery does not import its Python, expose hooks/tools/skills, install a driver or start services. Backend activation is profile-scoped; its desktop UI has a separate opt-in toggle in Settings → Plugins. Existing conversations keep their cached schemas until a new session.

## Setup

The runtime requires Linux, a working systemd user session, labwc, Xwayland, WayVNC, D-Bus/AT-SPI, grim, wlr-randr, bubblewrap and a render node. Install system prerequisites explicitly with your distribution's package manager. The driver installer supports **Linux x86-64 only**. Unsupported platforms must leave native Realms disabled.

```sh
hermes plugins enable hermes-realms
hermes realms --help
hermes realms install-driver
hermes realms doctor
```

Use `hermes -p NAME` for a named profile. `install-driver` is an explicit download of the pinned, checksum-verified driver into that profile's `plugin-data/hermes-realms/bin/`; it never modifies global binaries or the packaged source. `--archive /path/to/release.tar.gz` uses an existing verified archive without downloading. Enabling the plugin does not run the installer, edit Hyprland config or install system services.

Restart the owning backend after changing Python activation, then start a new conversation. Enable **Realms** separately in desktop Plugins settings for badges, Watch and Pop out. A UI toggle alone does not activate backend hooks.

## Session controls

- `/realm on` selects a private desktop; startup is lazy on the next eligible tool.
- `/realm status` reports mode and readiness; `/realm size 1280x720` resizes it.
- `/realm watch` opens a short-lived view-only viewer; human takeover pauses agent input.
- `/realm stop` stops owned processes without selecting the host route.
- `/realm off` returns tools to the ordinary host route **only when explicitly requested**.

After opt-in, the default mode is `realm`. Set `hermes config set plugins.realms.default_mode ask` to require a choice first. Realm failures never authorize a host-display/input fallback; normal Hermes approvals still apply. Remote Watch/Pop out is unsupported without an explicit viewer tunnel.

To disable, stop active realms, turn off its desktop UI, run `hermes plugins disable hermes-realms`, and restart that profile's backend. Profile data and the explicitly installed driver are retained.

The plugin's source README documents adopted-source provenance, notices, architecture and verification limits. Loader/artifact tests alone are not proof of live compositor, private input or native window behavior.
