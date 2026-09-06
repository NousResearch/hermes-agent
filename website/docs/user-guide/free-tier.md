---
sidebar_position: 3
title: "Free tier and signing in"
description: "What Hermes gives you before you add a key or sign in, how the free tier coexists with your own API key, how to sign in, and how to turn it off."
---

# Free tier and signing in

A fresh Hermes install works before you paste an API key or sign in anywhere. The first time you
run a command, Hermes sets up the **Nous free tier** (a few seconds, shown as "Setting up free
inference…") and answers on the `nous/welcome` model. Nothing to configure, no wizard to click
through. `hermes setup` is still there when you want it; it is never forced.

## What you get out of the box

| | Free tier | After signing in |
|---|---|---|
| Inference | `nous/welcome` (one model) | Full Nous Portal catalog |
| Connectors (Gmail, Linear, Notion, ...) | Yes | Yes |
| Paid tools through the [Tool Gateway](/user-guide/features/tool-gateway) (web search, image generation, TTS, cloud browser) | No | Yes, billed to your subscription |
| Credits or a balance | None | Yes |

"Connectors" are the third-party accounts you link on the Nous portal so the agent can act in
them. They work on the free tier without any sign-in.

While the free tier carries inference, the banner and `hermes auth status` read
`Nous · free tier · nous/welcome`, and `hermes model` lists a **Nous · free tier** row with that
single model. Asking for another model on the free tier prints a pointer instead of switching
silently:

```text
gpt-5 needs a Nous account or an API key. Run `hermes auth upgrade` or `hermes model`.
```

Calling a paid tool prints `This tool requires a Nous account. Run hermes auth upgrade.` and the
turn continues without it.

If `model.default` in `config.yaml` names something other than `nous/welcome` while the free tier
is doing inference, Hermes uses `nous/welcome` anyway and says so in one line. The free tier
serves exactly one model.

## Using your own API key alongside it

The free tier is the last resort, never a preference. Any provider you configure wins:

| You have | Inference runs on | Connectors |
|---|---|---|
| Nothing | Nous free tier (`nous/welcome`) | Free tier |
| An API key in `.env` (OpenRouter, OpenAI, Anthropic, ...) | Your key | Free tier |
| `model.provider` set in `config.yaml` | That provider | Free tier |
| A Nous Portal sign-in | Nous Portal | Your account |

On an install that already has a provider, the free tier sets itself up quietly in the
background on the next start so connectors have something to authenticate with, and prints a
one-time notice:

```text
Free Nous inference and connectors are now available. `hermes model` to try them, `hermes auth upgrade` to sign in.
```

You can pick the free tier explicitly from `hermes model` (or `/model`) like any other provider.

## Signing in

```bash
hermes auth upgrade
```

The command name is provisional and may change in a later release; the behaviour will not.

1. Hermes prints a URL and a short code, and opens the browser unless you pass `--no-browser`
   or you are in an SSH session. Never share the code.
2. Sign in to Nous Portal in the browser and confirm.
3. Back in the terminal: `Signed in as you@example.com. Your connectors are kept.`
   If your default model was `nous/welcome`, a fourth line names the model your account now
   uses, for example `Default model is now upstage/solar-pro4:free.`

Connectors you linked on the free tier carry over. Inference moves to the Nous Portal catalog,
paid tools unlock, and `hermes auth status` shows your account instead of the free-tier line.
`nous/welcome` stays with the free tier: an account that was using it lands on the recommended
model for its plan (the same one a fresh `hermes model` pick would suggest), and a default model
you chose yourself is left alone.

`hermes auth upgrade` is offered wherever the free tier is present, including installs that
run inference on their own API key. Signing in still unlocks paid tools for those installs.

:::note Plain login starts fresh
`hermes auth add nous --type oauth` also signs you in, but it replaces the free tier outright and
does not carry your connectors over. Use `hermes auth upgrade` when you have connectors you want
to keep.
:::

## Turning the free tier off

```bash
hermes config set nous.guest false
```

`nous.guest` is a normal `config.yaml` setting (default `true`), not an environment variable.
With it off:

| | `nous.guest: true` (default) | `nous.guest: false` |
|---|---|---|
| Free inference on `nous/welcome` | Available | Off |
| Connectors without sign-in | Available | Off |
| Free-tier row in `hermes model` | Shown | Hidden |
| Fresh install with nothing configured | Chats immediately | Offered `hermes setup` |
| Signing in with a Nous account | Works | Works |

Nothing else changes. A signed-in Nous account, your own API keys, and every other provider work
exactly as before. Set it back to `true` and the free tier returns on the next command that
needs it.

## What `hermes logout` does

| Situation | Result |
|---|---|
| Only the free tier is present | Nothing is cleared. Hermes prints: `You're not signed in. Free inference and connectors are always on. Run hermes auth to sign in with a Nous account.` |
| Signed in with a Nous account | The sign-in is removed from this profile and from the shared store, so no other profile on this machine picks it back up. With `nous.guest: true` the install returns to the free tier the next time it needs inference or a connector. |
| Another provider is active | Unchanged behaviour: that provider's stored credential is cleared. |

There is no command to reset or recreate the free tier. It is created once and looks after
itself.

## Troubleshooting

| Symptom | What it means | What to do |
|---|---|---|
| First command prints `It looks like Hermes isn't configured yet` and offers `hermes setup` | The free tier could not be set up within a few seconds: you are offline, or the free tier is not open on the portal Hermes is pointed at, or it is rate limited. | Come back online and run the command again, or run `hermes setup` and add a provider of your own. Nothing is left half-configured. |
| `Nous free tier is not open on this portal.` | The portal Hermes is pointed at is not offering the free tier right now. If you set `HERMES_PORTAL_BASE_URL`, that portal may not have it at all. | Sign in with an account, unset a portal override you no longer need, or add your own key with `hermes setup`. |
| `Nous free tier is rate limited; try again shortly.` | The portal is throttling new free-tier setups at the moment. | Wait a few minutes and retry, or add your own key with `hermes setup`. |
| `This tool requires a Nous account.` | You called a paid Tool Gateway tool on the free tier. | `hermes auth upgrade`, or configure that tool with your own key in `hermes tools`. |
| Model picker shows only `nous/welcome` under Nous | Expected on the free tier. | Sign in for the full catalog, or add an API key for another provider. |
| The free tier stopped working after two weeks away | The free-tier identity expired (see below) and is replaced on next use. | Nothing; run any command. Connectors linked before the gap need to be linked again unless you had signed in. |

## Privacy

To make the free tier work, Hermes creates an identity on the Nous portal the first time it
needs one and stores the credential in your Hermes directory, shared across the profiles under
that directory. That identity holds no email address, no name, and no other personal data; it
exists so inference and connector calls can be authenticated and rate limited. It expires after
14 days without use, at which point Hermes transparently creates a new one the next time you run
a command. Signing in with `hermes auth upgrade` moves what that identity holds (your linked
connectors) into your account. Turning the free tier off with `nous.guest: false` means no
identity is created or used at all.
