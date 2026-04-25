---
title: "Sherlock — OSINT username search across 400+ social networks"
sidebar_label: "Sherlock"
description: "OSINT username search across 400+ social networks"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sherlock

OSINT username search across 400+ social networks. Hunt down social media accounts by username.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/security/sherlock` |
| Path | `optional-skills/security/sherlock` |
| Version | `1.0.0` |
| Author | unmodeled-tyler |
| License | MIT |
| Tags | `osint`, `security`, `username`, `social-media`, `reconnaissance` |

## Reference: full SKILL.md

:::info
The following is the complete skill definition that Hermes loads when this skill is triggered. This is what the agent sees as instructions when the skill is active.
:::

# Sherlock OSINT Username Search

Hunt down social media accounts by username across 400+ social networks using the [Sherlock Project](https://github.com/sherlock-project/sherlock).

## When to Use

- User asks to find accounts associated with a username
- User wants to check username availability across platforms
- User is conducting OSINT or reconnaissance research
- User asks "where is this username registered?" or similar

## Requirements

- Sherlock CLI installed: `pipx install sherlock-project` or `pip install sherlock-project`
- Alternatively: Docker available (`docker run -it --rm sherlock/sherlock`)
- Network access to query social platforms

## Procedure

### 1. Check if Sherlock is Installed

**Before doing anything else**, verify sherlock is available:

```bash
sherlock --version
```

If the command fails:
- Offer to install: `pipx install sherlock-project` (recommended) or `pip install sherlock-project`
- **Do NOT** try multiple installation methods — pick one and proceed
- If installation fails, inform the user and stop

### 2. Extract Username

**Extract the username directly from the user's message if clearly stated.**

Examples where you should **NOT** use clarify:
- "Find accounts for nasa" → username is `nasa`
- "Search for johndoe123" → username is `johndoe123`
- "Check if alice exists on social media" → username is `alice`
- "Look up user bob on social networks" → username is `bob`

**Only use clarify if:**
- Multiple potential usernames mentioned ("search for alice or bob")
- Ambiguous phrasing ("search for my username" without specifying)
- No username mentioned at all ("do an OSINT search")

When extracting, take the **exact** username as stated — preserve case, numbers, underscores, etc.

### 3. Build Command

**Default command** (use this unless user specifically requests otherwise):
```bash
sherlock --print-found --no-color "<username>" --timeout 90
```

**Optional flags** (only add if user explicitly requests):
- `--nsfw` — Include NSFW sites (only if user asks)
- `--tor` — Route through Tor (only if user asks for anonymity)

**Do NOT ask about options via clarify** — just run the default search. Users can request specific options if needed.

### 4. Execute Search

Run via the `terminal` tool. The command typically takes 30-120 seconds depending on network conditions and site count.

**Example terminal call:**
```json
{
  "command": "sherlock --print-found --no-color \"target_username\"",
  "timeout": 180
}
```

### 5. Parse and Present Results

Sherlock outputs found accounts in a simple format. Parse the output and present:

1. **Summary line:** "Found X accounts for username 'Y'"
2. **Categorized links:** Group by platform type if helpful (social, professional, forums, etc.)
3. **Output file location:** Sherlock saves results to `<username>.txt` by default

**Example output parsing:**
```
[+] Instagram: https://instagram.com/username
[+] Twitter: https://twitter.com/username
[+] GitHub: https://github.com/username
```

Present findings as clickable links when possible.

## Pitfalls

### No Results Found
If Sherlock finds no accounts, this is often correct — the username may not be registered on checked platforms. Suggest:
- Checking spelling/variation
- Trying similar usernames with `?` wildcard: `sherlock "user?name"`
- The user may have privacy settings or deleted accounts

### Timeout Issues
Some sites are slow or block automated requests. Use `--timeout 120` to increase wait time, or `--site` to limit scope.

### Tor Configuration
`--tor` requires Tor daemon running. If user wants anonymity but Tor isn't available, suggest:
- Installing Tor service
- Using `--proxy` with an alternative proxy

### False Positives
Some sites always return "found" due to their response structure. Cross-reference unexpected results with manual checks.

### Rate Limiting
Aggressive searches may trigger rate limits. For bulk username searches, add delays between calls or use `--local` with cached data.

## Installation

### pipx (recommended)
```bash
pipx install sherlock-project
```

### pip
```bash
pip install sherlock-project
```

### Docker
```bash
docker pull sherlock/sherlock
docker run -it --rm sherlock/sherlock <username>
```

### Linux packages
Available on Debian 13+, Ubuntu 22.10+, Homebrew, Kali, BlackArch.

## Ethical Use

This tool is for legitimate OSINT and research purposes only. Remind users:
- Only search usernames they own or have permission to investigate
- Respect platform terms of service
- Do not use for harassment, stalking, or illegal activities
- Consider privacy implications before sharing results

## Verification

After running sherlock, verify:
1. Output lists found sites with URLs
2. `<username>.txt` file created (default output) if using file output
3. If `--print-found` used, output should only contain `[+]` lines for matches

## Example Interaction

**User:** "Can you check if the username 'johndoe123' exists on social media?"

**Agent procedure:**
1. Check `sherlock --version` (verify installed)
2. Username provided — proceed directly
3. Run: `sherlock --print-found --no-color "johndoe123" --timeout 90`
4. Parse output and present links

**Response format:**
> Found 12 accounts for username 'johndoe123':
>
> • https://twitter.com/johndoe123
> • https://github.com/johndoe123
> • https://instagram.com/johndoe123
> • [... additional links]
>
> Results saved to: johndoe123.txt

---

**User:** "Search for username 'alice' including NSFW sites"

**Agent procedure:**
1. Check sherlock installed
2. Username + NSFW flag both provided
3. Run: `sherlock --print-found --no-color --nsfw "alice" --timeout 90`
4. Present results
