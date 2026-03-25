---
sidebar_position: 3
title: "Updating & Uninstalling"
description: "How to update Hermes Agent to the latest version or uninstall it"
---

# Updating & Uninstalling

## Updating

Update to the latest version with a single command:

```bash
hermes update
```

This pulls the latest code, updates dependencies, and prompts you to configure any new options that were added since your last update.

:::tip
`hermes update` automatically detects new configuration options and prompts you to add them. If you skipped that prompt, you can manually run `hermes config check` to see missing options, then `hermes config migrate` to interactively add them.
:::

### Recommended Post-Update Validation

`hermes update` handles the main update path, but it does not run a full last-mile smoke test for your local environment. After updating, run this short validation flow:

1. `hermes update`
2. `git status --short`
3. If the tree is unexpectedly dirty, inspect that before doing anything else.
4. `hermes doctor`
5. `hermes --version` or `python run_agent.py --help`
6. If you use the gateway: `hermes gateway status`
7. If you use WhatsApp: pay attention to bridge-specific `doctor` output
8. If `doctor` reports high/critical npm issues: run `npm audit fix` in the flagged directory
9. If the update touched submodules, or behavior still seems off: `git submodule update --init --recursive`

:::warning Dirty working tree after update
If `git status --short` shows unexpected changes after `hermes update`, stop and inspect them before continuing. This usually means either local changes were reapplied on top of the updated codebase, or a dependency step refreshed generated files such as lockfiles.
:::

### Important Caveats

- `hermes update` updates the main repo and reinstalls dependencies, but it does not currently refresh Git submodules. If upstream changes submodule SHAs, run:

  ```bash
  git submodule update --init --recursive
  ```

- `hermes update` runs `npm install` for the repo root. If you use WhatsApp, note that the installers also manage the separate `scripts/whatsapp-bridge` Node environment. `hermes doctor` audits both locations, which is why it belongs in the standard post-update flow.

### Updating from Messaging Platforms

You can also update directly from Telegram, Discord, Slack, or WhatsApp by sending:

```
/update
```

This pulls the latest code, updates dependencies, and restarts the gateway.

### Manual Update

If you installed manually (not via the quick installer):

```bash
cd /path/to/hermes-agent
export VIRTUAL_ENV="$(pwd)/venv"

# Pull latest code and submodules
git pull origin main
git submodule update --init --recursive

# Reinstall (picks up new dependencies)
uv pip install -e ".[all]"
uv pip install -e "./tinker-atropos"

# Check for new config options
hermes config check
hermes config migrate   # Interactively add any missing options
```

---

## Uninstalling

```bash
hermes uninstall
```

The uninstaller gives you the option to keep your configuration files (`~/.hermes/`) for a future reinstall.

### Manual Uninstall

```bash
rm -f ~/.local/bin/hermes
rm -rf /path/to/hermes-agent
rm -rf ~/.hermes            # Optional — keep if you plan to reinstall
```

:::info
If you installed the gateway as a system service, stop and disable it first:
```bash
hermes gateway stop
# Linux: systemctl --user disable hermes-gateway
# macOS: launchctl remove ai.hermes.gateway
```
:::
