#!/usr/bin/env bash
# Hermes Bot Desktop — one headless Xfce desktop per Hermes profile, served over RFB.
#
# Spawned by tools/bot_desktop/runtime.py with HERMES_BD_* variables set. Runs TigerVNC's Xvnc
# (X server + RFB server in one process; damage-driven, resizable via SetDesktopSize) listening on a
# 0600 Unix socket only, then a minimal Xfce started component-wise under a private dbus session.
#
# Why not startxfce4 / xfce4-session: xfce4-session expects a logind session scope; outside one it
# spawns polkit agents that pop empty dialogs and light-locker/xfce4-screensaver lock the desktop for a
# user who has no password. Starting xfsettingsd -> xfwm4 -> xfdesktop -> xfce4-panel directly, with
# the screensaver/locker/power-manager autostarts masked, is the shape every headless-VNC recipe
# converges on (TigerVNC #1096/#581, OpenOnDemand's apptainer desktop, the Arch wiki).
#
# Why not the xfce4 metapackage: it drags in the screensaver, power manager and polkit agent this
# script exists to keep out.
set -euo pipefail

: "${HERMES_BD_PROFILE:?}"      # profile name (display name in the VNC title)
: "${HERMES_BD_DISPLAY_NUM:?}"  # allocated by runtime.py
: "${HERMES_BD_SOCKET:?}"       # RFB unix socket path
: "${HERMES_BD_XAUTH:?}"        # Xauthority path
: "${HERMES_BD_ENV_FILE:?}"     # where to publish DISPLAY/XAUTHORITY/DBUS_SESSION_BUS_ADDRESS
: "${HERMES_BD_CONFIG_HOME:?}"  # per-profile XDG_CONFIG_HOME (xfconf lives here)
GEOM="${HERMES_BD_GEOMETRY:-1440x900}"
DEPTH=24

export XDG_CONFIG_HOME="$HERMES_BD_CONFIG_HOME"
export XDG_CACHE_HOME="${HERMES_BD_CACHE_HOME:-$HERMES_BD_CONFIG_HOME/.cache}"
export XDG_DATA_HOME="${HERMES_BD_DATA_HOME:-$HERMES_BD_CONFIG_HOME/.local-share}"
export XDG_SESSION_TYPE=x11 XDG_CURRENT_DESKTOP=XFCE
export GDK_BACKEND=x11 QT_QPA_PLATFORM=xcb NO_AT_BRIDGE=1 GTK_A11Y=none
export LANG="${LANG:-C.UTF-8}"
# Inheriting a login session's bus/session manager yields "Another session manager is already
# running" / "Unable to contact settings server".
unset SESSION_MANAGER DBUS_SESSION_BUS_ADDRESS DISPLAY XAUTHORITY WAYLAND_DISPLAY

mkdir -p "$XDG_CONFIG_HOME/xfce4/xfconf/xfce-perchannel-xml" "$XDG_CONFIG_HOME/autostart" \
         "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$(dirname "$HERMES_BD_SOCKET")"

export DISPLAY=":$HERMES_BD_DISPLAY_NUM"
export XAUTHORITY="$HERMES_BD_XAUTH"

# Stale lock files from a crashed server block restart; a lock whose pid is alive belongs to a
# running server (another profile may have taken this number) and is never touched — Xvnc then
# fails to start on it and runtime.py reports that instead of us disrupting the other desktop.
rm -f "$HERMES_BD_SOCKET"
xlock="/tmp/.X${HERMES_BD_DISPLAY_NUM}-lock"
if [[ -e "$xlock" ]] && ! kill -0 "$(tr -d ' ' < "$xlock" 2>/dev/null)" 2>/dev/null; then
  rm -f "$xlock" "/tmp/.X11-unix/X${HERMES_BD_DISPLAY_NUM}"
fi
: > "$XAUTHORITY"; chmod 600 "$XAUTHORITY"
cookie="$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
printf 'add %s MIT-MAGIC-COOKIE-1 %s\n' "$DISPLAY" "$cookie" | xauth -q -f "$XAUTHORITY"
unset cookie

# ---- look: dark theme from whatever the host ships (first match wins), Hermes wallpaper ----
pick_theme() { local d t; for t in "$@"; do for d in /usr/share/themes "$HOME/.themes"; do [[ -d "$d/$t" ]] && { echo "$t"; return; }; done; done; echo "$1"; }
pick_icons() { local d t; for t in "$@"; do for d in /usr/share/icons "$HOME/.icons"; do [[ -d "$d/$t" ]] && { echo "$t"; return; }; done; done; echo "$1"; }
GTK_THEME_NAME=$(pick_theme Adwaita-dark Breeze-Dark Greybird-dark Arc-Dark Adwaita)
WM_THEME_NAME=$(pick_theme Default-hdpi Default)   # xfwm4 window themes ship with xfwm4 itself
ICON_THEME_NAME=$(pick_icons Papirus-Dark breeze-dark Adwaita hicolor)
: "${HERMES_BD_WALLPAPER:="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/wallpaper.png"}"

# ---- pre-seed xfconf BEFORE xfconfd starts (it caches; edits after start are overwritten) ----
X="$XDG_CONFIG_HOME/xfce4/xfconf/xfce-perchannel-xml"
[[ -e "$X/xfwm4.xml" ]] || cat > "$X/xfwm4.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfwm4" version="1.0">
  <property name="general" type="empty">
    <property name="use_compositing" type="bool" value="false"/>
    <property name="workspace_count" type="int" value="1"/>
    <property name="focus_new" type="bool" value="true"/>
    <property name="theme" type="string" value="HERMES_BD_WM_THEME"/>
    <property name="title_font" type="string" value="DejaVu Sans Bold 9"/>
  </property>
</channel>
EOF
sed -i "s|HERMES_BD_WM_THEME|$WM_THEME_NAME|" "$X/xfwm4.xml"
[[ -e "$X/xfce4-screensaver.xml" ]] || cat > "$X/xfce4-screensaver.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-screensaver" version="1.0">
  <property name="saver" type="empty"><property name="enabled" type="bool" value="false"/></property>
  <property name="lock" type="empty"><property name="enabled" type="bool" value="false"/></property>
</channel>
EOF
[[ -e "$X/xsettings.xml" ]] || cat > "$X/xsettings.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xsettings" version="1.0">
  <property name="Net" type="empty">
    <property name="EnableEventSounds" type="bool" value="false"/>
    <property name="ThemeName" type="string" value="HERMES_BD_GTK_THEME"/>
    <property name="IconThemeName" type="string" value="HERMES_BD_ICON_THEME"/>
  </property>
  <property name="Gtk" type="empty">
    <property name="FontName" type="string" value="DejaVu Sans 10"/>
    <property name="MonospaceFontName" type="string" value="DejaVu Sans Mono 10"/>
  </property>
  <property name="Xft" type="empty">
    <property name="DPI" type="int" value="96"/>
    <property name="Antialias" type="int" value="1"/>
    <property name="Hinting" type="int" value="1"/>
  </property>
</channel>
EOF
sed -i "s|HERMES_BD_GTK_THEME|$GTK_THEME_NAME|; s|HERMES_BD_ICON_THEME|$ICON_THEME_NAME|" "$X/xsettings.xml"
[[ -e "$X/xfce4-desktop.xml" ]] || cat > "$X/xfce4-desktop.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-desktop" version="1.0">
  <property name="backdrop" type="empty">
    <property name="screen0" type="empty">
      <property name="monitorVNC-0" type="empty">
        <property name="workspace0" type="empty">
          <property name="color-style" type="int" value="0"/>
          <property name="image-style" type="int" value="5"/>
          <property name="last-image" type="string" value="HERMES_BD_WALLPAPER_PLACEHOLDER"/>
          <property name="rgba1" type="array">
            <value type="double" value="0.11"/><value type="double" value="0.12"/>
            <value type="double" value="0.16"/><value type="double" value="1"/>
          </property>
        </property>
      </property>
    </property>
  </property>
  <property name="desktop-icons" type="empty">
    <property name="style" type="int" value="0"/>
  </property>
</channel>
EOF
sed -i "s|HERMES_BD_WALLPAPER_PLACEHOLDER|$HERMES_BD_WALLPAPER|" "$X/xfce4-desktop.xml"
# Own panel layout (a layout on disk also suppresses the first-run "Welcome to the panel" dialog):
# top bar = menu · tasks · tray · clock; bottom dock = only launchers whose program exists on this
# host, the browser pinned to the one the bot drives so a human lands in the bot's own browser profile.
if [[ ! -e "$X/xfce4-panel.xml" ]]; then
  L="$XDG_CONFIG_HOME/xfce4/panel"; mkdir -p "$L"
  dock_ids=(); n=20
  add_launcher() {  # name icon exec — skipped when the executable is missing
    local exe; exe=${3%% *}
    command -v "$exe" >/dev/null 2>&1 || return 0
    n=$((n+1)); mkdir -p "$L/launcher-$n"
    printf '[Desktop Entry]\nVersion=1.0\nType=Application\nName=%s\nIcon=%s\nExec=%s\nTerminal=false\nStartupNotify=false\n' \
      "$1" "$2" "$3" > "$L/launcher-$n/hermes.desktop"
    dock_ids+=("$n")
  }
  add_launcher "Terminal" utilities-terminal "xfce4-terminal"
  # The bot's browser: runtime.py resolves the executable agent-browser drives plus the profile's
  # persistent user-data-dir, so a human taking over lands in the bot's own cookie jar.
  [[ -n "${HERMES_BD_BROWSER_EXEC:-}" ]] && add_launcher "Browser" internet-web-browser "$HERMES_BD_BROWSER_EXEC"
  add_launcher "Files" system-file-manager "thunar"
  add_launcher "Text Editor" accessories-text-editor "mousepad"
  dock_plugins=""; dock_items=""
  for id in "${dock_ids[@]}"; do
    dock_plugins+="<value type=\"int\" value=\"$id\"/>"
    dock_items+="<property name=\"plugin-$id\" type=\"string\" value=\"launcher\"><property name=\"items\" type=\"array\"><value type=\"string\" value=\"hermes.desktop\"/></property></property>"
  done
  cat > "$X/xfce4-panel.xml" <<PANEL
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-panel" version="1.0">
  <property name="configver" type="int" value="2"/>
  <property name="panels" type="array">
    <value type="int" value="1"/>
    <value type="int" value="2"/>
    <property name="dark-mode" type="bool" value="true"/>
    <property name="panel-1" type="empty">
      <property name="position" type="string" value="p=6;x=0;y=0"/>
      <property name="length" type="uint" value="100"/>
      <property name="position-locked" type="bool" value="true"/>
      <property name="size" type="uint" value="28"/>
      <property name="icon-size" type="uint" value="16"/>
      <property name="background-style" type="uint" value="1"/>
      <property name="background-rgba" type="array">
        <value type="double" value="0.06"/><value type="double" value="0.07"/>
        <value type="double" value="0.09"/><value type="double" value="0.92"/>
      </property>
      <property name="plugin-ids" type="array">
        <value type="int" value="1"/><value type="int" value="2"/><value type="int" value="3"/>
        <value type="int" value="4"/><value type="int" value="5"/>
      </property>
    </property>
    <property name="panel-2" type="empty">
      <property name="position" type="string" value="p=10;x=0;y=0"/>
      <property name="length" type="uint" value="1"/>
      <property name="length-adjust" type="bool" value="true"/>
      <property name="position-locked" type="bool" value="true"/>
      <property name="size" type="uint" value="44"/>
      <property name="icon-size" type="uint" value="28"/>
      <property name="background-style" type="uint" value="1"/>
      <property name="background-rgba" type="array">
        <value type="double" value="0.06"/><value type="double" value="0.07"/>
        <value type="double" value="0.09"/><value type="double" value="0.80"/>
      </property>
      <property name="plugin-ids" type="array">${dock_plugins}</property>
    </property>
  </property>
  <property name="plugins" type="empty">
    <property name="plugin-1" type="string" value="applicationsmenu">
      <property name="show-button-title" type="bool" value="false"/>
    </property>
    <property name="plugin-2" type="string" value="tasklist">
      <property name="grouping" type="bool" value="true"/>
      <property name="show-labels" type="bool" value="true"/>
    </property>
    <property name="plugin-3" type="string" value="separator">
      <property name="expand" type="bool" value="true"/>
      <property name="style" type="uint" value="0"/>
    </property>
    <property name="plugin-4" type="string" value="systray"/>
    <property name="plugin-5" type="string" value="clock">
      <property name="digital-time-format" type="string" value="%a %H:%M"/>
    </property>
    ${dock_items}
  </property>
</channel>
PANEL
fi
# Mask system autostarts that want logind/polkit/keyring/at-spi.
for a in xfce4-screensaver light-locker xfce4-power-manager xfce-polkit \
         polkit-gnome-authentication-agent-1 lxpolkit xfce4-notifyd blueman at-spi-dbus-bus \
         gnome-keyring-pkcs11 gnome-keyring-secrets gnome-keyring-ssh xdg-user-dirs; do
  [[ -e "$XDG_CONFIG_HOME/autostart/$a.desktop" ]] || \
    printf '[Desktop Entry]\nType=Application\nName=%s\nHidden=true\n' "$a" > "$XDG_CONFIG_HOME/autostart/$a.desktop"
done
# Tests seed the config tree on a fake PATH and stop here (no X server needed).
[[ -n "${HERMES_BD_SEED_ONLY:-}" ]] && exit 0

# ---- X server + RFB (TigerVNC Xvnc), Unix socket only ----
# SecurityTypes None is safe ONLY because -rfbport -1 disables TCP and the 0600 socket is reachable
# only by processes running as this user (the gateway's WebSocket bridge does the real authentication;
# same-UID processes, the bot's own terminal tool included, are inside that boundary by design).
# -SendCutText=0: watchers must never receive the holder's clipboard; -AcceptCutText stays on so
# paste INTO the screen keeps working.
Xvnc "$DISPLAY" -geometry "$GEOM" -depth "$DEPTH" -dpi 96 \
  -rfbport -1 -rfbunixpath "$HERMES_BD_SOCKET" -rfbunixmode 0600 \
  -SecurityTypes None -AlwaysShared -AcceptSetDesktopSize -FrameRate 30 -SendCutText=0 \
  -desktop "hermes:$HERMES_BD_PROFILE" -auth "$XAUTHORITY" -nolisten tcp \
  -Log '*:stderr:30' 2> >(grep -v --line-buffered 'Could not resolve keysym' >&2) &
XVNC_PID=$!
trap 'kill "$XVNC_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 100); do
  xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break
  kill -0 "$XVNC_PID" 2>/dev/null || { echo "Xvnc exited during startup" >&2; exit 1; }
  sleep 0.1
done
xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 || { echo "Xvnc did not become ready" >&2; exit 1; }

setxkbmap -display "$DISPLAY" us 2>/dev/null || true   # RFB keysyms + xdotool assume a known layout
xsetroot -display "$DISPLAY" -solid '#1c1f29' 2>/dev/null || true
xset -display "$DISPLAY" s off -dpms s noblank 2>/dev/null || true

# ---- private session bus + Xfce components (no xfce4-session) ----
# dbus-run-session scopes the bus to this subshell: no leaked dbus-daemons on restart. The env file
# is written from INSIDE the bus so DBUS_SESSION_BUS_ADDRESS is the real one; runtime.py and every
# cua-driver / browser spawn for this profile source it.
# Not exec'd: this script stays the supervisor so the EXIT trap above still reaps Xvnc when the Xfce
# session dies on its own (exec would replace the trap's owner and orphan the X server).
dbus-run-session -- bash -c '
  set -e
  umask 077
  printf "DISPLAY=%s\nXAUTHORITY=%s\nDBUS_SESSION_BUS_ADDRESS=%s\nXDG_CONFIG_HOME=%s\nXDG_CACHE_HOME=%s\nXDG_DATA_HOME=%s\n" \
    "$DISPLAY" "$XAUTHORITY" "$DBUS_SESSION_BUS_ADDRESS" "$XDG_CONFIG_HOME" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" \
    > "$HERMES_BD_ENV_FILE.tmp" && mv -f "$HERMES_BD_ENV_FILE.tmp" "$HERMES_BD_ENV_FILE"
  xfsettingsd --sm-client-disable --daemon 2>/dev/null || true
  xfwm4 --compositor=off --sm-client-disable &
  for _ in $(seq 1 50); do xprop -root _NET_SUPPORTING_WM_CHECK >/dev/null 2>&1 && break; sleep 0.1; done
  xfdesktop --sm-client-disable --disable-wm-check &
  exec xfce4-panel --sm-client-disable --disable-wm-check
' && rc=0 || rc=$?
kill "$XVNC_PID" 2>/dev/null || true
exit "$rc"
