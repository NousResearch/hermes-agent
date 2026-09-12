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

# Stale lock files from a crashed server block restart.
rm -f "$HERMES_BD_SOCKET" "/tmp/.X${HERMES_BD_DISPLAY_NUM}-lock" "/tmp/.X11-unix/X${HERMES_BD_DISPLAY_NUM}"
: > "$XAUTHORITY"; chmod 600 "$XAUTHORITY"
xauth -q -f "$XAUTHORITY" add "$DISPLAY" MIT-MAGIC-COOKIE-1 "$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"

# ---- pre-seed xfconf BEFORE xfconfd starts (it caches; edits after start are overwritten) ----
X="$XDG_CONFIG_HOME/xfce4/xfconf/xfce-perchannel-xml"
[[ -e "$X/xfwm4.xml" ]] || cat > "$X/xfwm4.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfwm4" version="1.0">
  <property name="general" type="empty">
    <property name="use_compositing" type="bool" value="false"/>
    <property name="workspace_count" type="int" value="1"/>
    <property name="focus_new" type="bool" value="true"/>
  </property>
</channel>
EOF
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
  </property>
  <property name="Xft" type="empty">
    <property name="DPI" type="int" value="96"/>
    <property name="Antialias" type="int" value="1"/>
    <property name="Hinting" type="int" value="1"/>
  </property>
</channel>
EOF
[[ -e "$X/xfce4-desktop.xml" ]] || cat > "$X/xfce4-desktop.xml" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<channel name="xfce4-desktop" version="1.0">
  <property name="backdrop" type="empty">
    <property name="screen0" type="empty">
      <property name="monitorVNC-0" type="empty">
        <property name="workspace0" type="empty">
          <property name="color-style" type="int" value="0"/>
          <property name="image-style" type="int" value="0"/>
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
# The vendor default panel layout suppresses the first-run "Welcome to the panel" dialog.
if [[ ! -e "$X/xfce4-panel.xml" ]]; then
  for d in /etc/xdg/xfce4/panel/default.xml /usr/share/xfce4-panel/default.xml \
           /etc/xdg/xdg-xubuntu/xfce4/panel/default.xml; do
    [[ -e "$d" ]] && { cp "$d" "$X/xfce4-panel.xml"; break; }
  done
fi
# Mask system autostarts that want logind/polkit/keyring/at-spi.
for a in xfce4-screensaver light-locker xfce4-power-manager xfce-polkit \
         polkit-gnome-authentication-agent-1 lxpolkit xfce4-notifyd blueman at-spi-dbus-bus \
         gnome-keyring-pkcs11 gnome-keyring-secrets gnome-keyring-ssh xdg-user-dirs; do
  [[ -e "$XDG_CONFIG_HOME/autostart/$a.desktop" ]] || \
    printf '[Desktop Entry]\nType=Application\nName=%s\nHidden=true\n' "$a" > "$XDG_CONFIG_HOME/autostart/$a.desktop"
done

# ---- X server + RFB (TigerVNC Xvnc), Unix socket only ----
# SecurityTypes None is safe ONLY because -rfbport -1 disables TCP and the 0600 socket is reachable
# solely by the gateway process, whose WebSocket bridge performs the real authentication.
Xvnc "$DISPLAY" -geometry "$GEOM" -depth "$DEPTH" -dpi 96 \
  -rfbport -1 -rfbunixpath "$HERMES_BD_SOCKET" -rfbunixmode 0600 \
  -SecurityTypes None -AlwaysShared -AcceptSetDesktopSize -FrameRate 30 \
  -desktop "hermes:$HERMES_BD_PROFILE" -auth "$XAUTHORITY" -nolisten tcp \
  -Log '*:stderr:30' &
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
exec dbus-run-session -- bash -c '
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
'
