#!/usr/bin/env bash
# One-time preparation of a fresh Ubuntu LTS VPS for this stack (spec 6.1-6.2). Idempotent:
# running it again changes only what is missing or wrong, so it doubles as a "repair" script.
#
#   scp infra/scripts/bootstrap-server.sh root@<ip>:   # or curl it from GitHub
#   ssh root@<ip> 'DEPLOY_PUBKEY="ssh-ed25519 AAAA... github-actions" bash bootstrap-server.sh'
#   (add ADMIN_PUBKEY="<your own public key>" if the provider gave you a root password only)
#
# What it does, in order:
#   1. packages: git, ufw, openssl, unattended-upgrades (security patches install themselves);
#   2. Docker Engine + the compose plugin from Docker's own apt repository (Ubuntu's docker.io
#      lags behind; docker-compose.prod.yml needs Compose >= 2.24 for `!reset`);
#   3. user `deploy`: runs the stack and is the SSH user of the deploy workflow. It is in the
#      `docker` group, which is root-equivalent (it can mount / into a container) - the point
#      is not to have CI log in as root, not to sandbox it. It gets root's authorized_keys
#      (the admin can log in as deploy) plus DEPLOY_PUBKEY (the key GitHub Actions uses);
#   4. firewall: incoming 22, 80, 443 only (spec 6.2). NOTE: Docker's published ports bypass
#      ufw, which is why docker-compose.prod.yml publishes only Caddy on 0.0.0.0;
#   5. SSH: keys only, no passwords, root only with a key. Applied only if root's
#      authorized_keys holds a key other than DEPLOY_PUBKEY (the provider's, or ADMIN_PUBKEY),
#      otherwise this would lock the owner out: `deploy` has no sudo, and the CI key is not
#      a way for a human to get root back. sshd -t checks the result before the reload;
#   6. 2 GB swap: insurance against the OOM killer at a training peak, not working memory;
#   7. the repository cloned to APP_DIR (owned by deploy; deploy.sh checks out versions);
#   8. .env from .env.example with every `change-me` replaced by a generated secret, the public
#      name <ip>.sslip.io and COOKIE_SECURE=true. An existing .env is never overwritten;
#   9. nightly backup at 03:30 UTC (infra/scripts/backup.sh) as deploy.
# At the end it prints what to put into the GitHub secrets.
#
# Environment (optional): DEPLOY_USER (deploy), APP_DIR (/opt/movielens-recsys), REPO_URL,
#   SITE_ADDRESS (default <primary IPv4>.sslip.io), DEPLOY_PUBKEY, ADMIN_PUBKEY,
#   SWAP_SIZE (2G, 0 = none).

set -euo pipefail

DEPLOY_USER=${DEPLOY_USER:-deploy}
APP_DIR=${APP_DIR:-/opt/movielens-recsys}
REPO_URL=${REPO_URL:-https://github.com/mark37mur-ops/recommended_system_pipeline_petproject.git}
SWAP_SIZE=${SWAP_SIZE:-2G}
# Overridable for the tests only (tests/unit/test_bootstrap_ssh.py).
ROOT_KEYS=${ROOT_KEYS:-/root/.ssh/authorized_keys}
SSHD_DROPIN_DIR=${SSHD_DROPIN_DIR:-/etc/ssh/sshd_config.d}
SSH_KEYGEN=${SSH_KEYGEN:-ssh-keygen}

log() { printf 'bootstrap: %s\n' "$*" >&2; }

# ----- secrets ------------------------------------------------------------------------------
# 32 random bytes as URL-safe base64 without "=" padding (43 characters), like Python's
# secrets.token_urlsafe(32): safe inside the database URLs compose builds from these.
gen_token() {
  openssl rand -base64 32 | tr '+/' '-_' | tr -d '=\n'
}

# Airflow's Fernet key must be URL-safe base64 of exactly 32 bytes WITH padding (44 chars).
gen_fernet_key() {
  openssl rand -base64 32 | tr '+/' '-_' | tr -d '\n'
}

# generate_env <.env.example> <.env> <site address>
# Copies the template line by line: comments stay, `KEY=change-me` gets a fresh secret,
# SITE_ADDRESS and COOKIE_SECURE get their production values. Never touches an existing file.
generate_env() {
  local example=$1 target=$2 site=$3 line key value
  if [ -e "$target" ]; then
    log "$target exists, left as it is"
    return 0
  fi
  # Written to a temporary file and renamed at the end: an interrupted run (Ctrl-C, a dropped
  # SSH session) must not leave half a .env that the next run would keep as "exists".
  (
    umask 077  # the file holds every secret of the stack
    while IFS= read -r line || [ -n "$line" ]; do
      # Comments, blank lines and anything that is not KEY=value are copied as they are.
      if [[ ! $line =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
        printf '%s\n' "$line"
        continue
      fi
      key=${BASH_REMATCH[1]}
      value=${BASH_REMATCH[2]}
      case $key in
        AIRFLOW_FERNET_KEY) value=$(gen_fernet_key) ;;
        SITE_ADDRESS) value=$site ;;
        COOKIE_SECURE) value=true ;;
        *) [ "$value" = change-me ] && value=$(gen_token) ;;
      esac
      printf '%s=%s\n' "$key" "$value"
    done < "$example" > "$target.tmp"
    mv -f "$target.tmp" "$target"
  )
  log "$target created with generated secrets"
}

# ----- system steps (root) --------------------------------------------------------------------
install_packages() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q ca-certificates curl git gnupg openssh-client openssl ufw unattended-upgrades
  # Install security updates automatically (Ubuntu's default policy: security pocket only).
  printf 'APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Unattended-Upgrade "1";\n' \
    > /etc/apt/apt.conf.d/20auto-upgrades
}

install_docker() {
  if docker compose version > /dev/null 2>&1; then
    log "docker with compose plugin already installed: $(docker compose version --short)"
    return 0
  fi
  # Docker's apt repository, key pinned to a file (signed-by), as in docs.docker.com.
  local codename
  # shellcheck source=/dev/null  # a system file, not part of the repository
  codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc]" \
    "https://download.docker.com/linux/ubuntu $codename stable" > /etc/apt/sources.list.d/docker.list
  apt-get update -q
  apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
}

create_user() {
  local home keys
  if ! id "$DEPLOY_USER" > /dev/null 2>&1; then
    adduser --disabled-password --gecos "" "$DEPLOY_USER"
  fi
  usermod -aG docker "$DEPLOY_USER"
  home=$(getent passwd "$DEPLOY_USER" | cut -d: -f6)
  keys="$home/.ssh/authorized_keys"
  install -d -m 700 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$home/.ssh"
  touch "$keys"
  # Add each key once (re-runs do not duplicate lines).
  {
    [ -f "$ROOT_KEYS" ] && cat "$ROOT_KEYS"
    [ -n "${DEPLOY_PUBKEY:-}" ] && printf '%s\n' "$DEPLOY_PUBKEY"
    true
  } | while IFS= read -r key; do
    [ -n "$key" ] && ! grep -qxF "$key" "$keys" && printf '%s\n' "$key" >> "$keys"
    true
  done
  chown "$DEPLOY_USER:$DEPLOY_USER" "$keys"
  chmod 600 "$keys"
}

setup_firewall() {
  # Default deny incoming; SSH, HTTP (ACME challenge + redirect), HTTPS on tcp and udp (HTTP/3).
  ufw default deny incoming
  ufw default allow outgoing
  ufw allow 22/tcp
  ufw allow 80/tcp
  ufw allow 443
  ufw --force enable
}

# valid_pubkey <line>: ONE line that ssh-keygen accepts as a public key.
valid_pubkey() {
  local tmp ok=1
  command -v "$SSH_KEYGEN" > /dev/null || return 1
  case $1 in *$'
'*) return 1 ;; esac
  tmp=$(mktemp)
  printf '%s
' "$1" > "$tmp"
  "$SSH_KEYGEN" -l -f "$tmp" > /dev/null 2>&1 || ok=0
  rm -f "$tmp"
  [ "$ok" = 1 ]
}

# ADMIN_PUBKEY (optional): the owner's own key, added to root's authorized_keys once. A key
# ssh-keygen rejects is refused before anything changes: it would count as "an admin can get
# in" and the next step would switch password login off.
add_admin_key() {
  [ -n "${ADMIN_PUBKEY:-}" ] || return 0
  valid_pubkey "$ADMIN_PUBKEY" ||
    { log "ERROR: ADMIN_PUBKEY is not one valid public key line (checked with $SSH_KEYGEN -l)"; return 1; }
  [ -d "$(dirname "$ROOT_KEYS")" ] || install -d -m 700 "$(dirname "$ROOT_KEYS")"
  touch "$ROOT_KEYS"
  chmod 600 "$ROOT_KEYS"
  grep -qxF "$ADMIN_PUBKEY" "$ROOT_KEYS" || printf '%s
' "$ADMIN_PUBKEY" >> "$ROOT_KEYS"
}

# How many keys let a HUMAN in as root: lines of root's authorized_keys that ssh-keygen accepts
# as a public key, other than the CI key (DEPLOY_PUBKEY). Comments, blanks and junk do not
# count. The CI key does not count either: its private half lives in GitHub Secrets and
# `deploy` has no sudo, so with only that key nobody could get root back. Without ssh-keygen
# nothing can be validated: 0, i.e. password login stays on (fail safe).
admin_key_count() {
  local ci line n=0
  if [ ! -f "$ROOT_KEYS" ] || ! command -v "$SSH_KEYGEN" > /dev/null; then
    echo 0
    return 0
  fi
  ci=$(printf '%s
' "${DEPLOY_PUBKEY:-}" | awk '{ print $2 }')
  while IFS= read -r line || [ -n "$line" ]; do
    [[ $line =~ ^[[:space:]]*(#|$) ]] && continue
    [ -n "$ci" ] && [[ $line == *"$ci"* ]] && continue
    if valid_pubkey "$line"; then n=$((n + 1)); fi
  done < "$ROOT_KEYS"
  echo "$n"
}

harden_ssh() {
  local conf=$SSHD_DROPIN_DIR/00-hardening.conf
  if [ "$(admin_key_count)" -eq 0 ]; then
    log "WARNING: no admin key in $ROOT_KEYS (the CI key DEPLOY_PUBKEY does not count):"
    log "password login left ON. Add your own key (ssh-copy-id root@<ip>, or ADMIN_PUBKEY=...)"
    log "and run this script again"
    return 0
  fi
  # sshd uses the FIRST value it reads; cloud images ship 50-cloud-init.conf with
  # PasswordAuthentication yes, so this file has to sort before it.
  cat > "$conf" <<'EOF'
# Written by infra/scripts/bootstrap-server.sh: SSH keys only (spec 6.2).
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
EOF
  # A config sshd rejects must reach neither the running server nor the next restart.
  if ! sshd -t; then
    rm -f "$conf"
    log "ERROR: sshd rejects the configuration: hardening removed again, nothing reloaded"
    return 1
  fi
  systemctl reload ssh
}

setup_swap() {
  [ "$SWAP_SIZE" != 0 ] || return 0
  if swapon --show=NAME --noheadings | grep -q .; then
    log "swap already active"
    return 0
  fi
  fallocate -l "$SWAP_SIZE" /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  # Prefer dropping page cache over swapping out the running services.
  sysctl -w vm.swappiness=10
  echo 'vm.swappiness=10' > /etc/sysctl.d/90-swappiness.conf
}

clone_repo() {
  if [ -d "$APP_DIR/.git" ]; then
    log "$APP_DIR already cloned (deploy.sh moves it between versions)"
  else
    install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$APP_DIR"
    sudo -u "$DEPLOY_USER" git clone --quiet "$REPO_URL" "$APP_DIR"
  fi
}

default_site_address() {
  # Address of the interface that routes to the internet: on a VPS that is the public IP.
  local ip
  ip=$(ip -4 route get 1.1.1.1 | awk '{ for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1) }')
  [ -n "$ip" ] || { log "cannot detect the public IP: set SITE_ADDRESS"; exit 1; }
  echo "$ip.sslip.io"
}

setup_env() {
  local site=${SITE_ADDRESS:-}
  [ -n "$site" ] || site=$(default_site_address)
  generate_env "$APP_DIR/.env.example" "$APP_DIR/.env" "$site"
  chown "$DEPLOY_USER:$DEPLOY_USER" "$APP_DIR/.env"
}

setup_backup_cron() {
  cat > /etc/cron.d/movielens-backup <<EOF
# Nightly backup (infra/scripts/backup.sh); written by bootstrap-server.sh.
30 3 * * * $DEPLOY_USER $APP_DIR/infra/scripts/backup.sh >> $APP_DIR/backups/backup.log 2>&1
EOF
  install -d -m 700 -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$APP_DIR/backups"
}

print_next_steps() {
  local host
  host=$(awk -F='$1 == "SITE_ADDRESS" { print $2 }' "$APP_DIR/.env")
  host=${host%.sslip.io}
  cat <<EOF

Done. GitHub -> Settings -> Secrets and variables -> Actions -> New repository secret:
  DEPLOY_HOST         $host
  DEPLOY_USER         $DEPLOY_USER
  DEPLOY_SSH_KEY      the PRIVATE key whose public half is DEPLOY_PUBKEY
  DEPLOY_KNOWN_HOSTS  the lines below (this server's host keys, read from disk, not scanned):
EOF
  for f in /etc/ssh/ssh_host_*_key.pub; do
    printf '%s %s\n' "$host" "$(cut -d' ' -f1,2 "$f")"
  done
  echo
  echo "Then push to main (or Actions -> Deploy -> Run workflow) to start the stack."
}

main() {
  [ "$(id -u)" -eq 0 ] || { log "run as root"; exit 1; }
  install_packages
  install_docker
  add_admin_key
  create_user
  setup_firewall
  harden_ssh
  setup_swap
  clone_repo
  setup_env
  setup_backup_cron
  print_next_steps
}

# Sourcing the file (tests) defines the functions without running anything.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
