# Hermes Agent — Security Advisory: Mini Shai-Hulud worm (mistralai 2.4.6)

**Date:** May 12, 2026
**Status:** Quarantined upstream / mitigated in Hermes
**Severity:** Critical
**Affected:** Users who installed `hermes-agent[all]` or `hermes-agent[mistral]` between the upload of `mistralai 2.4.6` and PyPI's quarantine of the package.

## What happened

The Mini Shai-Hulud supply-chain worm crossed from npm to PyPI on 2026-05-12.
Among the compromised PyPI artifacts was `mistralai 2.4.6` — the official
Mistral AI Python SDK. The worm steals credentials from environment
variables and credential files (`~/.npmrc`, `~/.pypirc`, `~/.aws/credentials`,
GitHub PATs, cloud SDK tokens) and exfils them to a hardcoded webhook.

Hermes Agent listed `mistralai>=2.3.0,<3` as the runtime dependency for its
optional Mistral TTS / STT providers. Users who installed
`pip install -e ".[all]"` between the malicious upload and the quarantine
pulled `mistralai 2.4.6` into their venv. PyPI has since removed the project
(`pypi:project-status: quarantined`), so the package is no longer
installable, but copies that landed before quarantine remain in users'
environments.

## Am I affected?

Run on the host where you installed Hermes:

```bash
hermes doctor
```

If the **Security Advisories** section flags
`mistralai==2.4.6`, you have the compromised package and must remediate.
If it flags any **other** version of `mistralai`, you are not on the
compromised release — but we still recommend uninstalling, since the
project is currently quarantined and we have disabled Mistral TTS / STT
in Hermes regardless.

You can also check manually:

```bash
pip show mistralai 2>/dev/null | grep -i version
```

## What we've done in Hermes Agent

1. **Removed `mistral` from the `[all]` extra** so fresh installs no
   longer pull the package by default. (PR #24205, already on main.)
2. **Disabled the Mistral TTS and STT providers** in the runtime — they
   return a "temporarily disabled" error and won't import the SDK even
   if the venv still has it.
3. **Added a security advisory checker** (`hermes doctor` and CLI startup
   banner) that detects `mistralai 2.4.6` if it's still installed and
   surfaces remediation steps. The banner is rate-limited (max once per
   24h per advisory) and dismissible via `hermes doctor --ack`.
4. **Hardened the installer fallback tiers.** When one extra's
   dependency becomes unavailable on PyPI, the installer now degrades
   gracefully — keeping every other extra — instead of dropping all the
   way to a stripped install. Future supply-chain incidents won't
   silently demote users.
5. **Added a lazy-install framework** (`tools/lazy_deps.py`) so opt-in
   backends (Mistral, ElevenLabs, Honcho, etc.) can be installed on
   demand when the user enables them, rather than eagerly at install
   time. This shrinks every fresh install's blast radius for future
   single-package compromises.

## What you should do

If `hermes doctor` flags `mistralai==2.4.6`, treat the credentials in
your environment as exposed:

1. **Uninstall the compromised package:**
   ```bash
   pip uninstall -y mistralai
   # or, if you installed via uv:
   uv pip uninstall mistralai
   ```

2. **Rotate API keys.** Every key in `~/.hermes/.env` should be rotated:
   OpenRouter, Anthropic, OpenAI, Nous, GitHub, AWS, Google, Mistral,
   and any other provider tokens you have configured. If you used a
   shell that exported keys (`.bashrc`, `.zshrc`, etc.), rotate those
   too.

3. **Audit credential files** for tokens that may have been read:
   `~/.npmrc`, `~/.pypirc`, `~/.aws/credentials`, `~/.config/gh/hosts.yml`,
   `~/.docker/config.json`, `~/.kube/config`, `~/.ssh/`. The worm
   harvested files matching these patterns.

4. **Check GitHub** for unexpected new SSH keys, deploy keys, or webhook
   additions on repositories you have admin on. The worm uses stolen
   GitHub tokens to add backdoors.

5. **After cleanup**, dismiss the Hermes warning:
   ```bash
   hermes doctor --ack shai-hulud-2026-05
   ```

## When will Mistral TTS / STT come back?

When PyPI restores the `mistralai` project to a clean release and we
verify the new release on a clean network, we will re-enable Mistral
TTS / STT in Hermes Agent. Until then, use Edge TTS (default, no key),
ElevenLabs, OpenAI TTS, MiniMax TTS, or any of the user-defined command
providers. For STT, use Groq Whisper or OpenAI Whisper.

## Future hardening

This incident exposed two structural weaknesses in our install path:

- Eager-install of every optional extra meant ONE compromised package
  could break the whole `[all]` resolve. **Fixed** via tiered fallback +
  lazy-install framework.
- Users had no way to know whether they had a poisoned dependency.
  **Fixed** via `hermes_cli/security_advisories.py` and the
  `hermes doctor` integration.

We will continue to extend `tools/lazy_deps.py` so additional opt-in
backends (Slack, Matrix, Bedrock, DingTalk, Feishu, Google Workspace,
YouTube transcripts, etc.) can be installed on first use rather than
eagerly. This reduces the blast radius of any future single-package
compromise.

## References

- Socket Security report: <https://socket.dev/blog/mini-shai-hulud-worm-pypi>
- PyPI quarantine: <https://pypi.org/simple/mistralai/>
  (project-status: quarantined as of 2026-05-12)
- Hermes Agent PR (mistral disabled): #24205
- Hermes Agent PR (advisory checker + lazy installs): _this PR_
- GitHub security advisory: _to be filed alongside this PR_

## Credits

Reported via [@SocketSecurity](https://twitter.com/SocketSecurity) and
the broader supply-chain security community. Hermes Agent's response
(detection, lazy-install framework, installer tier hardening) was built
by the Hermes Agent team at Nous Research.
