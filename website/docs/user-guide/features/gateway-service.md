# Gateway Service

The messaging gateway (Telegram, Discord, WhatsApp, and every other
platform adapter) normally runs as a child of wherever you started it —
a terminal, `hermes serve`, or the desktop app's backend. On Windows,
the **bundled MSIX install** ships one more way: a real Windows Service.

## What it is

Installing the desktop app registers a service named **HermesGateway**
in the Windows Service Control Manager. It arrives **stopped** and
**demand-start** — it never starts on its own. When you turn it on, the
Service Control Manager owns the gateway's lifecycle:

- **automatic at logon** — bots start when you sign in, no app needed
- **restart on crash** — the SCM respawns the gateway if it dies
- **bots survive app close** — the service is not the desktop app's
  child; closing the app never touches it

The service runs as *you* (the installing user), using your Hermes home
— the same sessions, memory, and config as every other surface. The
default profile's gateway is what it serves; multiplexed profiles ride
the same process, exactly as a foreground gateway.

## Turning it on

```bash
hermes gateway service on
```

That flips the service to automatic-at-logon and starts it now,
gracefully. Turning it off is the same shape:

```bash
hermes gateway service off      # demand-start + graceful stop
hermes gateway service status   # SCM state + the config key
```

The posture persists in the `gateway.service` config key, so `hermes
gateway service status` is the one command that answers "is this on?"

## Updates and the graceful stop

Stopping the service — by you, by an update, or by Windows shutdown —
is always graceful: in-flight agent sessions drain, the session database
flushes, and platform locks release (the same drain path `hermes
gateway stop` uses). `hermes update` on a service-managed install
restarts the service through the SCM (`Restart-Service`) — the drain
happens first, the respawn after, so a mixed-version gateway is never
left serving.

## Uninstall

Uninstalling the desktop app removes the service automatically (MSIX
registers it; MSIX removes it). Bots stop — same as today, where
uninstalling the payload removes the gateway's runtime.

## Platform notes

- **Windows MSIX installs only.** Source installs (and every other
  platform) keep the foreground `hermes gateway run`, systemd, or
  launchd paths — see [the CLI gateway
  docs](/user-guide/cli#gateway).
- The service requires Windows 10 1903+ (the MSIX's minimum OS — the
  same floor the desktop app ships with).
- `services.msc` shows it as "Hermes Gateway"; manage it there if you
  like, though the `hermes gateway service` verbs keep the config key
  in sync, which the SCM console does not.
