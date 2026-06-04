---
sidebar_position: 0
title: "Run Nemotron 3 Ultra free in Hermes Agent"
description: "Try NVIDIA Nemotron 3 Ultra on Nous Portal — free June 4–18 — with day 0 support in Hermes Agent"
---

# Run Nemotron 3 Ultra free in Hermes Agent

Nous Research has been inducted into the **Nemotron Coalition** of leading AI labs working with **NVIDIA** to advance open frontier foundation models. In honor of this, we've partnered with **Nebius** to provide **Nemotron 3 Ultra** free on [Nous Portal](https://portal.nousresearch.com) for two weeks (**June 4th – June 18th**). Follow the instructions below to try the model in your Hermes Agent today.

:::info Limited-time offer
The `nvidia/nemotron-3-ultra:free` tier is available from **June 4th to June 18th**. The `:free` tag is what keeps it on the no-cost plan — pick that exact variant.
:::

## Before you start

You'll need a terminal (macOS, Linux, or Windows via [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)) with `curl` installed. `curl` is preinstalled on most systems.

## Steps

### 1. Install Hermes Agent

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Prefer to review first? Download [`install.sh`](https://hermes-agent.nousresearch.com/install.sh), inspect it, then run it.

After it finishes, reload your shell:

```bash
source ~/.bashrc   # or source ~/.zshrc
```

### 2. Run Quick Setup

```bash
hermes setup
```

Select **Quick Setup**. Hermes opens a browser tab and waits for you to finish the next steps.

### 3. Create a Nous Portal account

In the browser, create a [Nous Portal](https://portal.nousresearch.com) account (or sign in) and choose the **Free** plan.

### 4. Connect your account

When prompted to connect your account to Hermes Agent, click **Connect**. You'll see a confirmation once it's linked.

### 5. Select the free Nemotron 3 Ultra model

Return to your terminal. From the model list, select:

```
nvidia/nemotron-3-ultra:free
```

The `:free` tag is what keeps it on the no-cost tier, so make sure you pick that variant.

### 6. Start chatting

Complete the remaining Quick Setup prompts, then run:

```bash
hermes
```

That's it — you're talking to Nemotron 3 Ultra, free.

## Switching to it later

Already set up with another model? Switch any time from inside a session:

```bash
/model nvidia/nemotron-3-ultra:free
```

Or open the picker and choose it from the list:

```bash
/model
```

## Troubleshooting

- **Don't see the model in the list?** Make sure you finished the Nous Portal connection (step 4) and that you're on the **Free** plan. Run `hermes portal info` to confirm you're logged in and routing through Nous.
- **Picked the wrong variant?** Re-select `nvidia/nemotron-3-ultra:free` with `/model` — the `:free` suffix is required to stay on the no-cost tier.
- **Browser didn't open / you're on a remote host?** See [OAuth over SSH / Remote Hosts](/guides/oauth-over-ssh) for port-forwarding and manual-paste workarounds.

## See also

- **[Run Hermes Agent with Nous Portal](/guides/run-hermes-with-nous-portal)** — Full Portal walkthrough: models, Tool Gateway, and verification
- **[Nous Portal integration](/integrations/nous-portal)** — What's in the subscription
- **[Quickstart](/getting-started/quickstart)** — Install-to-chat in under 5 minutes
