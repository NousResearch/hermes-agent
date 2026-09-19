---
sidebar_position: 8
title: Official Domains & Phishing Safety
description: How to verify you are using genuine Hermes Agent resources, and how to spot impersonation scams.
---

# Official Domains & Phishing Safety

Hermes Agent's popularity has attracted impersonation campaigns — fake "Hermes" websites, GitHub accounts that mass-mention contributors with shortlinks, and cold emails offering "credits" on lookalike products. This page lists the only official sources so you can verify anything that claims to be Hermes Agent.

## Official sources

These are the **only** official Hermes Agent domains and accounts:

- **GitHub repository:** [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)
- **Documentation & installer:** [hermes-agent.nousresearch.com](https://hermes-agent.nousresearch.com/)
- **Nous Portal (cloud & subscriptions):** [portal.nousresearch.com](https://portal.nousresearch.com/)
- **Nous Research website:** [nousresearch.com](https://nousresearch.com/)
- **X (Twitter):** [@NousResearch](https://x.com/NousResearch)

The install commands only ever come from `hermes-agent.nousresearch.com`:

```bash
# Linux / macOS / WSL2 / Termux
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

```powershell
# Windows (native)
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

If an installer, download, or "portal" is hosted anywhere else, treat it as untrusted.

## Not official

- **Any other domain containing "hermes-agent" or "hermes"** (e.g. `.icu`, `.org`, `.ai` variants) is not operated by Nous Research, even when it copies the branding or says it "points to official docs".
- **Cryptocurrency tokens.** There is **no Hermes Agent or Nous Research token**. Any account or site promoting one is a scam.
- **GitHub accounts that mass-mention users** in discussions or issues with "community notice" posts and URL-shortener links (`goo.su`, `bit.ly`, etc.). Genuine project announcements never arrive as mass @-mentions with shortlinks.
- **Unsolicited emails** offering credits, hosted "multiplayer Hermes" products, or partnership deals that name-drop the project to build trust.

## If you receive a suspicious message

1. **Don't click** shortened or off-domain links; don't enter credentials or connect wallets.
2. **Report the account** to GitHub (`Report abuse` on the profile) or the relevant platform.
3. **Check this page** — if a domain isn't listed above, it isn't ours.
4. Optionally flag it in the [Nous Research Discord](https://discord.gg/nousresearch) or an issue so others see it, but note that impersonation campaigns are external abuse — they're handled by platform reports, not code changes.

Hermes Agent is MIT-licensed and free to self-host. Nobody legitimate will ever ask you to pay for the open-source agent itself, claim "your account has credits waiting", or DM you an alternate download link.
