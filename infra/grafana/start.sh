#!/bin/sh
# Entrypoint wrapper of the Grafana container. Why: provisioning files are static YAML, but the
# Telegram contact point is optional, and Grafana refuses to start with a Telegram contact point
# whose bot token is empty. So the provisioning directory is assembled at start-up:
#   /etc/grafana/provisioning-repo   (read-only mount of infra/grafana/provisioning)
#   + /etc/grafana/telegram.yml      (only if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set)
#     or /etc/grafana/telegram-off.yml (otherwise: removes a contact point left from before)
#   -> /tmp/grafana-provisioning     (what Grafana reads, GF_PATHS_PROVISIONING)
# and then the image's own entrypoint (/run.sh) takes over.
set -eu

src=/etc/grafana/provisioning-repo
dst=/tmp/grafana-provisioning

rm -rf "$dst"
# Every folder Grafana scans, also the ones we do not use (it logs an error for a missing one).
mkdir -p "$dst/alerting" "$dst/dashboards" "$dst/datasources" "$dst/plugins"
cp -R "$src"/. "$dst"/

if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
  mkdir -p "$dst/alerting"
  cp /etc/grafana/telegram.yml "$dst/alerting/telegram.yml"
  echo "start.sh: Telegram contact point enabled"
else
  cp /etc/grafana/telegram-off.yml "$dst/alerting/telegram-off.yml"
  echo "start.sh: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set: no Telegram contact point," \
       "alerts are visible in the Grafana UI only"
fi

export GF_PATHS_PROVISIONING="$dst"
exec /run.sh "$@"
