**Skill runtime.conf paths break under MSYS bash on Windows.**

When a skill's `runtime.conf` uses a CLI path like `/d/Hermes/skills/anysearch/scripts/anysearch_cli.py`, MSYS bash translates the leading `/d/` to `D:\` and then appends the remaining path starting with `d/`, resulting in the incorrect path `D:\d\Hermes\skills\anysearch\scripts\anysearch_cli.py`. This breaks cron jobs and any headless skill invocation.

**Symptoms:** Cron jobs fail with `Connection error` or `RuntimeError: ...` even though the skill works fine in interactive sessions. The error message may not mention the path at all — the skill's CLI just can't find its own script.

**Fix:** Use Windows-native paths (with forward slashes) in `runtime.conf` instead of MSYS-style paths:

```ini
# ❌ Breaks under MSYS bash
Runtime: python
Command: python /d/Hermes/skills/anysearch/scripts/anysearch_cli.py

# ✅ Works everywhere
Runtime: python
Command: python D:/Hermes/skills/anysearch/scripts/anysearch_cli.py
```

Also prefer the Python runtime over Node.js on Windows, because Node.js shebang scripts (`#!/usr/bin/env node`) are misinterpreted by Windows Script Host (`cscript.exe`) and fail with `Invalid character` errors at line 1, character 1.

```ini
# ❌ Triggers Windows Script Host error
Runtime: Node.js
Command: node C:\Users\...\.codex\skills\anysearch\scripts\anysearch_cli.js

# ✅ Python runtime avoids shebang issues
Runtime: python
Command: python D:/Hermes/skills/anysearch/scripts/anysearch_cli.py
```

**Root cause:** When Hermes runs a cron job, the shell environment is an MSYS bash session. Absolute paths like `/d/...` or `/c/...` get double-translated by MSYS's path conversion — the leading `/d/` becomes `D:\` and the `d/` in the path body is kept, producing `D:\d\...`. Windows-native paths (`D:/...`) bypass this translation entirely.