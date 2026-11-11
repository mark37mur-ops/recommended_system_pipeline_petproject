#!/usr/bin/env bash
# One-time preparation of a fresh Ubuntu LTS VPS for this stack (spec 6.1-6.2). Idempotent:
# running it again changes only what is missing or wrong, so it doubles as a "repair" script.
#
#   scp infra/scripts/bootstrap-server.sh root@<ip>:   # or curl it from GitHub
#   ssh root@<ip> 'DEPLOY_PUBKEY="ssh-ed25519 AAAA... github-actions" bash bootstrap-server.sh'
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
#   5. SSH: keys only, no passwords, root only with a key. Applied only if somebody already
#      has an authorized key, otherwise this would lock the owner out;
#   6. 2 GB swap: insurance against the OOM killer at a training peak, not working memory;
#   7. the repository cloned to APP_DIR (owned by deploy; deploy.sh checks out versions);
#   8. .env from .env.example with every `change-me` replaced by a generated secret, the public
#      name <ip>.sslip.io and COOKIE_SECURE=true. An existing .env is never overwritten;
#   9. nightly backup at 03:30 UTC (infra/scripts/backup.sh) as deploy.
# At the end it prints what to put into the GitHub secrets.
#
# Environment (optional): DEPLOY_USER (deploy), APP_DIR (/opt/movielens-recsys), REPO_URL,
#   SITE_ADDRESS (default <primary IPv4>.sslip.io), DEPLOY_PUBKEY, SWAP_SIZE (2G, 0 = none).

set -euo pipefail

DEPLOY_USER=${DEPLOY_USER:-deploy}
APP_DIR=${APP_DIR:-/opt/movielens-recsys}
REPO_URL=${REPO_URL:-https://github.com/mark37mur-ops/recommended_system_pipeline_petproject.git}
SWAP_SIZE=${SWAP_SIZE:-2G}

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
    done < "$example" > "$target"
  )
  log "$target created with generated secrets"
}

# ----- system steps (root) --------------------------------------------------------------------
install_packages() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y -q ca-certificates curl git gnupg openssl ufw unattended-upgrades
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
    [ -f /root/.ssh/authorized_keys ] && cat /root/.ssh/authorized_keys
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

harden_ssh() {
  local home
  home=$(getent passwd "$DEPLOY_USER" | cut -d: -f6)
  if [ ! -s /root/.ssh/authorized_keys ] && [ ! -s "$home/.ssh/authorized_keys" ]; then
    log "WARNING: no authorized_keys for root or $DEPLOY_USER: password login left ON"
    log "add your key (ssh-copy-id) and run this script again"
    return 0
  fi
  # sshd uses the FIRST value it reads; cloud images ship 50-cloud-init.conf with
  # PasswordAuthentication yes, so this file has to sort before it.
  cat > /etc/ssh/sshd_config.d/00-hardening.conf <<'EOF'
# Written by infra/scripts/bootstrap-server.sh: SSH keys only (spec 6.2).
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
EOF
  sshd -t  # a broken config must not reach the running server
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
