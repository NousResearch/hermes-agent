---
title: "Hermes Tescmd Plugin"
sidebar_label: "Hermes Tescmd Plugin"
description: "Install and operate the native Hermes Tesla Fleet plugin for vehicle state and guarded Tesla Fleet API controls"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Tescmd Plugin

Install and operate the native Hermes Tesla Fleet plugin for vehicle state and guarded Tesla Fleet API controls.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/smart-home/hermes-tescmd-plugin` |
| Path | `optional-skills/smart-home/hermes-tescmd-plugin` |
| Version | `0.5.0a11` |
| Author | Oceanswave |
| License | MIT |
| Platforms | linux, macos |
| Tags | `tesla`, `fleet-api`, `smart-home`, `vehicle`, `hermes-plugin` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Hermes Tesla Fleet plugin

Use this skill when a user wants to install, configure, or operate `hermes-tescmd-plugin`, a standalone Hermes plugin that adds Tesla Fleet API operations to Hermes as native tools.

The plugin lets Hermes answer and act on requests such as checking charge state, warming the cabin, locking the vehicle, sending destinations to navigation, and reviewing Fleet readiness. It is designed for Hermes-native operation: install the package, restart Hermes so Python entry points reload, then use the registered `tescmd_*` tools.

## Install the plugin

Install the Python package into the same environment that runs Hermes:

```bash
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python hermes-tescmd-plugin
hermes gateway restart
```

For local editable development, install a checkout into Hermes' runtime venv:

```bash
uv pip install --python ~/.hermes/hermes-agent/venv/bin/python -e ~/hermes-tescmd-plugin
hermes gateway restart
```

## First-time setup

1. Create a Tesla Developer app that you control.
2. Configure Tesla app URLs using your public HTTPS domain:
   - Allowed Origin URL(s): `https://<your-domain>`
   - Allowed Redirect URI(s): `https://<your-domain>/callback`
   - Allowed Returned URL(s): leave blank unless Tesla requires it.
3. Create the plugin config file at:

```text
$HERMES_HOME/plugins/hermes-tescmd-plugin/config.json
```

4. Include your Tesla app client ID, optional client secret, region, callback/domain, requested scopes, and optional default VIN. Keep this app config in plugin-owned state, not in Hermes core `config.yaml`.
5. Run `tescmd_status` to see readiness booleans, missing prerequisites, derived callback/public-key URLs, and recommended next actions.
6. Start OAuth with `tescmd_auth_login`, open the returned Tesla authorization URL, then finish with `tescmd_auth_complete` using the callback URL or `code` + `state`.
7. For signed vehicle commands, generate and host a virtual-key public key with `tescmd_key_generate(confirm=true)` and `tescmd_key_deploy(method="local", confirm=true)`, then validate the hosted key before enrollment.

Read the package page for the complete public overview:

- https://pypi.org/project/hermes-tescmd-plugin/

## Daily operation

Prefer the native Hermes tool surface for agent work:

- `tescmd_status` for readiness and next steps.
- `tescmd_vehicle_list`, `tescmd_charge_status`, `tescmd_vehicle_location`, `tescmd_climate_status`, `tescmd_security_status`, and other read tools for state.
- `tescmd_charge_*`, `tescmd_climate_*`, `tescmd_security_*`, `tescmd_navigation_*`, `tescmd_media_*`, and `tescmd_vehicle_*` tools for operations.
- `tescmd_key_*` and `tescmd_auth_*` for admin/bootstrap flows.

## Safety model

Tesla operations can have real-world effects. Keep these rules:

- Side-effecting operations require `confirm: true` in tool arguments and must fail closed before network/file side effects when confirmation is missing.
- Waking a sleeping vehicle is a side effect; only set `wake=true` when explicitly requested or required.
- Prefer read tools before write tools when the target vehicle, readiness, or current state is uncertain.
- Do not paste OAuth tokens, client secrets, vehicle-command private keys, or exported auth blobs into chat.
- Plugin-owned operational state belongs under `$HERMES_HOME/plugins/hermes-tescmd-plugin/`.

## Troubleshooting

If the tools do not appear:

1. Verify the package is installed in the Hermes runtime environment.
2. Restart the Hermes CLI session or gateway so plugin entry points reload.
3. Run `tescmd_status` after reload to inspect app config, auth, key, and cache readiness.

If a side-effecting tool returns a confirmation error, retry with `confirm: true` only after verifying the requested action and target.
