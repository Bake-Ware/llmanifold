#!/bin/sh
# llmanifold installer. Installs into a private venv, writes a starter config
# and a systemd unit, and starts the service. Run it again to upgrade: the
# config and data are never overwritten.
#
#   curl -fsSL https://raw.githubusercontent.com/Bake-Ware/llmanifold/main/install.sh | sh
#
# As root it installs system-wide (/opt/llmanifold, /etc/llmanifold,
# /var/lib/llmanifold, user `llmanifold`). As anyone else it installs for that
# user (~/.local/share/llmanifold, ~/.config/llmanifold) with a systemd user unit.
#
# Options (also settable as environment variables):
#   --ref <ref>       branch, tag or commit to install   (LLMANIFOLD_REF, default main)
#   --source <path>   install from a local checkout instead of GitHub (LLMANIFOLD_SOURCE)
#   --no-service      skip the systemd unit               (LLMANIFOLD_NO_SERVICE=1)
#   --no-start        write the unit but don't start it   (LLMANIFOLD_NO_START=1)
#   --uninstall       remove the venv and unit; config and data are kept
set -eu

REPO="https://github.com/Bake-Ware/llmanifold"
REF="${LLMANIFOLD_REF:-main}"
SOURCE="${LLMANIFOLD_SOURCE:-}"
NO_SERVICE="${LLMANIFOLD_NO_SERVICE:-}"
NO_START="${LLMANIFOLD_NO_START:-}"
UNINSTALL=""

say() { printf '%s\n' "$*"; }
die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --ref) [ $# -ge 2 ] || die "--ref needs a value"; REF="$2"; shift 2 ;;
    --source) [ $# -ge 2 ] || die "--source needs a path"; SOURCE="$2"; shift 2 ;;
    --no-service) NO_SERVICE=1; shift ;;
    --no-start) NO_START=1; shift ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

if [ "$(id -u)" = 0 ]; then
  MODE=system
  PREFIX=/opt/llmanifold
  CONF_DIR=/etc/llmanifold
  DATA_DIR=/var/lib/llmanifold
  BIN_DIR=/usr/local/bin
  UNIT=/etc/systemd/system/llmanifold.service
  SYSTEMCTL="systemctl"
else
  MODE=user
  PREFIX="${XDG_DATA_HOME:-$HOME/.local/share}/llmanifold"
  CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/llmanifold"
  DATA_DIR="$PREFIX/data"
  BIN_DIR="$HOME/.local/bin"
  UNIT="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/llmanifold.service"
  SYSTEMCTL="systemctl --user"
fi
VENV="$PREFIX/venv"
CONFIG="$CONF_DIR/config.yaml"

have_systemd() { command -v systemctl >/dev/null 2>&1 && $SYSTEMCTL show-environment >/dev/null 2>&1; }

if [ -n "$UNINSTALL" ]; then
  if [ -f "$UNIT" ]; then
    if have_systemd; then $SYSTEMCTL disable --now llmanifold >/dev/null 2>&1 || true; fi
    rm -f "$UNIT"
    if have_systemd; then $SYSTEMCTL daemon-reload || true; fi
  fi
  [ -L "$BIN_DIR/llmanifold" ] && rm -f "$BIN_DIR/llmanifold"
  rm -rf "$VENV"
  say "llmanifold removed. Kept: $CONF_DIR and $DATA_DIR"
  exit 0
fi

# ---- python 3.11+
PY=""
for c in python3 python3.13 python3.12 python3.11; do
  if command -v "$c" >/dev/null 2>&1 &&
     "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || die "Python 3.11 or newer is required"

# ---- package
if [ -n "$SOURCE" ]; then
  [ -f "$SOURCE/pyproject.toml" ] || die "$SOURCE is not a llmanifold checkout"
  SPEC="$SOURCE"
else
  command -v git >/dev/null 2>&1 || die "git is required to install from GitHub"
  SPEC="git+$REPO@$REF"
fi

UPGRADE=""
[ -x "$VENV/bin/llmanifold" ] && UPGRADE=1

say "Installing llmanifold ($MODE) into $VENV"
mkdir -p "$PREFIX" "$CONF_DIR" "$DATA_DIR"
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" ||
  die "could not create a venv (on Debian/Ubuntu: apt install python3-venv)"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q --upgrade "$SPEC"
VERSION="$("$VENV/bin/llmanifold" --version)"

mkdir -p "$BIN_DIR"
ln -sf "$VENV/bin/llmanifold" "$BIN_DIR/llmanifold"

# ---- starter config (never overwritten)
if [ ! -f "$CONFIG" ]; then
  cat > "$CONFIG" <<EOF
# llmanifold config. The file is watched: endpoint and model changes apply
# without a restart. Every option: $REPO/blob/main/config.example.yaml

listen:
  api: 0.0.0.0:1234          # what clients point at
  admin: 127.0.0.1:1240      # dashboard + admin API; keep it private or behind an auth proxy

data_dir: $DATA_DIR
default_model: local

endpoints:
  local:
    url: http://127.0.0.1:8080   # your engine (llama.cpp, vLLM, Ollama, LM Studio, ...)
    max_concurrency: 1

models:
  local:
    aliases: [default]
    pool: [local]
EOF
  chmod 640 "$CONFIG"
  NEW_CONFIG=1
else
  NEW_CONFIG=""
fi
"$VENV/bin/llmanifold" check -c "$CONFIG" >/dev/null || die "config check failed: $CONFIG"

if [ "$MODE" = system ]; then
  id llmanifold >/dev/null 2>&1 ||
    useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin llmanifold
  chown -R llmanifold: "$DATA_DIR"
  # the admin site edits the config in place and keeps backups beside it
  chown -R llmanifold: "$CONF_DIR"
fi

# ---- systemd
STARTED=""
if [ -z "$NO_SERVICE" ] && have_systemd; then
  mkdir -p "$(dirname "$UNIT")"
  {
    say "[Unit]"
    say "Description=llmanifold LLM router"
    say "After=network-online.target"
    say "Wants=network-online.target"
    say ""
    say "[Service]"
    say "Type=simple"
    [ "$MODE" = system ] && say "User=llmanifold"
    say "WorkingDirectory=$DATA_DIR"
    say "EnvironmentFile=-$CONF_DIR/env"
    say "ExecStart=$VENV/bin/llmanifold serve -c $CONFIG"
    say "Restart=on-failure"
    say "RestartSec=2"
    say ""
    say "[Install]"
    if [ "$MODE" = system ]; then say "WantedBy=multi-user.target"; else say "WantedBy=default.target"; fi
  } > "$UNIT"
  $SYSTEMCTL daemon-reload
  if [ -z "$NO_START" ]; then
    if [ -n "$UPGRADE" ] && $SYSTEMCTL is-active --quiet llmanifold; then
      # a restart cuts requests in flight, so that stays the operator's call
      $SYSTEMCTL enable llmanifold >/dev/null 2>&1
      RESTART_HINT=1
    else
      $SYSTEMCTL enable --now llmanifold >/dev/null 2>&1
      STARTED=1
    fi
  fi
elif [ -z "$NO_SERVICE" ]; then
  say "systemd not available: skipping the service"
fi

say ""
say "$VERSION installed."
say "  config     $CONFIG"
say "  data       $DATA_DIR"
say "  command    $BIN_DIR/llmanifold"
[ -n "$NEW_CONFIG" ] && say "Edit the config to point at your engines (it starts with one at 127.0.0.1:8080)."
if [ -n "$STARTED" ]; then
  say "Running: API on :1234, dashboard on http://127.0.0.1:1240"
elif [ -n "${RESTART_HINT:-}" ]; then
  say "The running service is still the old version. When it is idle:"
  say "  $SYSTEMCTL restart llmanifold"
else
  say "Start it with: llmanifold serve -c $CONFIG"
fi
if [ "$MODE" = user ] && [ -n "$STARTED" ]; then
  say "To keep it running after you log out: sudo loginctl enable-linger $(id -un)"
fi
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "Note: $BIN_DIR is not on your PATH." ;; esac
