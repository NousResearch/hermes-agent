---
title: Bot Screen
sidebar_position: 17
---

# Bot Screen

On a headless Linux gateway host (a server, a cloud VM, Hermes Cloud) each bot
gets its **own desktop**: an Xfce screen the bot's `computer_use` and headed
browser act on, streamed live into Hermes Desktop. Watch what the bot does,
**take over** when it hits a login, 2FA prompt, CAPTCHA or payment step, then
**hand control back** and let it continue with the session you just signed in
to. The bot keeps working after you close the app or turn off your laptop; the
screen lives on the gateway host, not on your machine.

Every Hermes profile ("bot") has its own screen, its own browser profile and
its own cookies. Screens are work surfaces, not security boundaries: the bots
share the host's user account, files and network (the same model as other
hosted-agent products).

**Threat model.** The screen's RFB socket, the X display, the browser profile
and the control-lease file all belong to the gateway's OS user. Any process
running as that user — another bot on the same host, and the bot's own
`terminal` tool included — can reach them directly, bypassing the pane and the
lease. The lease is a tool-level fence on `computer_use` and the browser tools,
not an OS one. Running each bot as its own OS user is out of scope; if that
isolation matters to you, put the bots on separate hosts. Two timing details
worth knowing: the WebSocket bridge caches its lease decision for up to 250 ms
between re-reads of the lease file, so a takeover made by another process is
enforced within that window (the bot's tool results are voided by the lease
epoch regardless of the window). And the viewer's single-use, 30-second
`display_ticket` travels as a URL query parameter on purpose — noVNC cannot
negotiate WebSocket subprotocols, so a header is not an option — which means a
reverse proxy's access log may record an already-spent ticket.

## Requirements

- The gateway host runs Linux. macOS and Windows hosts already have a real
  display; the pane is not offered there.
- TigerVNC's `Xvnc` and the Xfce core components are installed on the host.
  Nothing installs them silently: `hermes update` and fresh installs leave every
  machine as it is. When they are missing the Screen pane in Hermes Desktop shows
  **Install on host** — one click runs the package manager on the gateway host
  (it asks for that host's sudo password in a masked card; the password goes to
  that host only and is never stored) and streams the log. When Hermes itself
  runs as root — the usual case in a container — the installer runs the package
  manager directly, with no sudo and no password card. When it is not root and
  the host has no `sudo` at all, the pane and the CLI print the exact install
  command for you to run on the host instead of showing a card. From a shell,
  `hermes computer-use screen status` prints the exact line and
  `hermes computer-use screen install` runs it:

  | Distro | Packages |
  |---|---|
  | Debian / Ubuntu | `tigervnc-standalone-server xfce4-panel xfwm4 xfdesktop4 xfce4-settings xfce4-terminal dbus-x11 x11-xserver-utils x11-utils xauth fonts-dejavu-core` |
  | Fedora | `tigervnc-x11-server xfce4-panel xfwm4 xfdesktop xfce4-settings xfce4-terminal dbus-daemon xsetroot xset xdpyinfo xprop xorg-x11-xauth setxkbmap dejavu-sans-fonts` |
  | Arch | `tigervnc xfce4-panel xfwm4 xfdesktop xfce4-settings xfce4-terminal xorg-xsetroot xorg-xset xorg-xdpyinfo xorg-xprop xorg-xauth xorg-setxkbmap ttf-dejavu` |

  On Fedora the `Xvnc` binary is in `tigervnc-x11-server` (not
  `tigervnc-server-minimal`) and `dbus-run-session` comes from `dbus-daemon`.
  Deliberately **not** the `xfce4` metapackage: it pulls in the screensaver,
  power manager and polkit agent that lock or prompt a headless desktop.
- [Computer Use](./computer-use.md) enabled for the bot (cua-driver installed).

## Using it

Every bot's computer is one click away in three places of Hermes Desktop:

- **Bots → a bot → Scheduled Jobs**: the bot's screen is the hero at the very
  top of the pane, above the title and the routines: a live preview of the
  desktop (refreshed every few seconds while the pane is visible) with who holds
  control; click the picture to expand into live access. While the screen is off
  or not installed the same box says so and offers Start / Install.
- **Bots → right-click a bot → Open Screen**.
- **Sessions sidebar**, grouped by gateway / profile: the same **Screen** box
  sits under each profile's header, so a profile's machine is reachable from
  its conversations too.

1. Open the Screen with any of the entries above.
   The screen is **off by default** and nothing starts it for you: click
   **Start screen** in the pane, run `hermes computer-use screen start` on the
   host, or set `bot_desktop.auto_start: true` if you want a headless host to
   start the screen by itself on the bot's first `computer_use` call (off so
   that installing TigerVNC never yields a screen nobody asked for). A headed
   browser opens on the screen once it is running.
2. The pane streams the bot's desktop. The chip in the header says who is in
   control: **Bot is in control** by default.
3. Click **Take over**. The border turns red, your keyboard and mouse now drive
   the bot's screen. Sign in, solve the CAPTCHA, approve the payment.
4. Click **Hand back**. The bot regains control and re-captures the screen
   before continuing. Closing the pane also hands control back. A *dropped*
   connection is different: if your laptop lid closes or Wi-Fi drops while you
   hold control, you keep it — the bot stays locked out of a screen you may be
   mid-login on — until you reconnect and hand back. If you come back after a
   reload and the pane still says a human holds control, a **Hand back (force)**
   button appears to clear it.

While you hold control, the bot's `computer_use` and browser tools are refused
with `human_has_control`, captures included. This is a tool-level fence, not an
OS one: the bot runs as the same user as its screen. Don't type secrets into a
bot you wouldn't trust with them.

When the bot hits a step it should not do itself (a login, 2FA, a CAPTCHA, a
payment) it says so in its reply and ends its turn; the ask reaches you in
whatever chat you are on. Take over when you are ready, do the step, hand back,
and tell the bot to continue. Nothing blocks on the bot's side while it waits:
taking over is always yours to start, and a bot never holds a tool call open
waiting for you.

Two viewers on one screen: the most recent **Take over** wins; the previous
controller drops back to watching.

## Browser sessions that survive the handoff

While the screen runs, the bot's browser tool and the dock's **Browser** icon are
the same browser: the Chromium agent-browser drives, with one persistent
user-data-dir per bot (`<HERMES_HOME>/bot-desktop/browser-profile`; set
`AGENT_BROWSER_PROFILE` to pin your own). Click Browser during a takeover and you
are in the bot's own windows and cookie jar; what you sign in to is what the bot
uses afterwards and in every later session, until the site expires the login.
Set `browser.headed: true` so the bot's own browsing is visible on the screen too.

The dock is seeded **once**, the first time the screen starts for a profile.
The guard is the panel layout file
`<HERMES_HOME>/bot-desktop/xdg/xfce4/xfconf/xfce-perchannel-xml/xfce4-panel.xml`:
while it exists the launcher leaves the panel alone, so changing
`AGENT_BROWSER_EXECUTABLE_PATH` or `AGENT_BROWSER_PROFILE` and restarting the
screen does not re-pin the Browser icon. Delete that file and the dock is
rebuilt on the next `screen start` from whatever is installed then.

Which Chromium the dock and the bot use: an explicit
`AGENT_BROWSER_EXECUTABLE_PATH` wins; otherwise Hermes prefers a system
`chromium` / `google-chrome` when one is installed, and falls back to the
Chromium Playwright bundled. The reason for that order is the sandbox: on
Ubuntu 23.10 and later, `kernel.apparmor_restrict_unprivileged_userns=1` stops
Playwright's bundled Chromium from setting up its sandbox for a non-root user
and it exits with `FATAL: No usable sandbox!`, while the distro's Chromium ships
with an AppArmor profile that allows it. If the pick is wrong for your host, set
`AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium` (or your Chrome path) in the
gateway's environment. The official Docker image ships only Playwright's
*headless shell*, which cannot draw a window, so inside it the dock has no
Browser icon and the pane / `screen status` report **no headed browser** until
you install a headed one (`apt-get install chromium`).

## CLI

```bash
hermes computer-use screen status          # installed? running? who holds control?
hermes computer-use screen start           # start this profile's screen
hermes computer-use screen stop            # stop it (hands control back first)
hermes computer-use screen install [-y]    # apt/dnf/pacman the packages
hermes -p research computer-use screen start   # another bot's screen
```

## Configuration

```yaml
bot_desktop:
  geometry: "1440x900"   # screen size; the viewer scales to fit the pane
  auto_start: false      # set true to start on the first computer_use call
```

`auto_start` is off by default. Start the screen from the Desktop's Screen
pane (**Start screen**), from `hermes computer-use screen start`, or set the
flag to `true` for a headless host that should bring its screen up the first
time the bot calls `computer_use` and no display is available.

State lives under `<HERMES_HOME>/bot-desktop/` per profile (RFB Unix socket,
Xauthority, launcher log, per-profile xfconf).

## How it works

- **TigerVNC `Xvnc`** is the X server and the RFB server in one process, per
  profile, listening only on a `0600` Unix socket. No TCP port, no VNC
  password: only processes running as the gateway's user can reach it (see the
  threat model above), and the gateway's WebSocket bridge is the authenticated
  way in.
- **Xfce** starts component-wise (`xfsettingsd`, `xfwm4 --compositor=off`,
  `xfdesktop`, `xfce4-panel`) under a private D-Bus session, without
  `xfce4-session`, so nothing tries to lock the screen or reach `logind`.
- **Hermes Desktop** bundles noVNC. It asks the gateway for a single-use ticket
  (`display.observe`) over its normal authenticated connection and opens a
  sibling WebSocket to `/api/display/ws`; the gateway splices the RFB stream
  through. Nothing new is exposed; the pane works over local, SSH, URL+token
  and Hermes Cloud connections alike.
- **Control lease.** The gateway drops keyboard, pointer and clipboard messages
  from any viewer that does not hold the lease, at the RFB byte level; noVNC's
  view-only flag is only the UI hint. The same lease gates `computer_use` and
  the browser tools. It is a file under `<HERMES_HOME>/bot-desktop/`: no file
  means the bot holds control (a fresh profile); a file that exists but cannot
  be read or parsed fails closed — the bot is treated as locked out until the
  next successful hand-off rewrites it. Xvnc never pushes the screen's clipboard
  to viewers (`-SendCutText=0`), so watchers do not receive what the person in
  control copies; pasting into the screen still works.
- **Display binding.** The launcher publishes `DISPLAY`, `XAUTHORITY` and the
  D-Bus address; every cua-driver and headed-browser spawn for that profile
  inherits them, so the bot never acts on a display a human is sitting at.

## Troubleshooting

- **"Screen packages missing"** — click **Install on host** in the pane, or run
  the printed install line on the gateway host (not on the machine running
  Hermes Desktop). The pane refuses a second install while one is running.
- **Screen starts then stops** — read `<HERMES_HOME>/bot-desktop/launcher.log`.
- **Typing produces wrong characters** — the screen uses a US keymap so RFB
  keysyms and cua-driver agree; change it with `setxkbmap` on that `DISPLAY`
  if you need another layout.
- **Bot says `human_has_control` after you left** — click **Hand back** in the
  pane (or **Hand back (force)** after a reload). From a shell,
  `hermes computer-use screen stop` releases the lease and stops the screen;
  `hermes computer-use screen start` brings it back with the bot in control.

### Testing under WSL

WSL2 counts as a supported Linux host: `screen status` reports it as such and
the pane is offered. One WSLg quirk gets in the way of the first start: WSLg
mounts `/tmp/.X11-unix` read-only, so `Xvnc` cannot create its display socket
and dies with `Cannot establish any listening sockets` in `launcher.log`.
Replace the mount with a writable directory before starting the screen:

```bash
sudo umount /tmp/.X11-unix
sudo mkdir -p /tmp/.X11-unix && sudo chmod 1777 /tmp/.X11-unix
```

The mount comes back on the next WSL restart; repeat the two commands then.
