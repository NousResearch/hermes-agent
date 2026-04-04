---
sidebar_position: 99
title: "Honcho Memory"
description: "Honcho is now available as a memory provider plugin"
---

# Honcho Memory

:::info Honcho is now a Memory Provider Plugin
Honcho has been integrated into the [Memory Providers](./memory-providers.md) system. All Honcho features are available through the unified memory provider interface.
:::

## Setup

```bash
hermes memory setup    # select "honcho"
```

Or set manually:

```yaml
# ~/.hermes/config.yaml
memory:
  provider: honcho
```

```bash
echo "HONCHO_API_KEY=your-key" >> ~/.hermes/.env
```

## Migrating from `hermes honcho`

If you previously used `hermes honcho setup`:

1. Your existing configuration (`honcho.json` or `~/.honcho/config.json`) is preserved
2. Your server-side data (memories, conclusions, user profiles) is intact
3. Just set `memory.provider: honcho` to reactivate

No re-login or re-setup needed. Run `hermes memory setup` and select "honcho" — the wizard detects your existing config.

## Full Documentation

See [Memory Providers — Honcho](./memory-providers.md#honcho) for tools, config reference, and details.
