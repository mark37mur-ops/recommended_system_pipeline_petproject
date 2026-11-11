#!/usr/bin/env bash
# Deploy one version of the stack on the server, roll back if it does not come up (spec 6.4).
#
#   infra/scripts/deploy.sh <image-tag> [git-ref]
#   infra/scripts/deploy.sh sha-1a2b3c4                  # git ref defaults to 1a2b3c4
#   infra/scripts/deploy.sh sha-1a2b3c4 1a2b3c4d5e6f...  # what .github/workflows/deploy.yml runs
#
# Steps:
#   1. remember the running version: IMAGE_TAG from .env and the checked-out commit;
#   2. check out the commit being deployed: docker-compose*.yml and the configs under infra/
#      (Caddyfile, Grafana provisioning, prometheus.yml) are read from this directory, so they
#      must match the code inside the images;
#   3. write the new IMAGE_TAG into .env, `docker compose pull`, `up -d --remove-orphans`;
#   4. wait until GET /ready answers 200 through Caddy (READY_TIMEOUT seconds): the API loads
#      the champion from MLflow after a restart and answers 503 until then. Not
#      `docker compose up --wait`: it fails on the one-shot jobs that exit by design;
#   5. failed pull, failed up or no /ready in time -> put the previous tag and commit back,
#      `up -d` again, wait for /ready again and exit 1 (the workflow turns red);
#   6. success -> delete images of this project other than the new, the previous and `latest`
#      (every version is up to ~1 GB of new layers; the previous one stays for a fast rollback).
#
# Environment (all optional):
#   APP_DIR          checkout with .env (default: two levels above this script)
#   READY_URL        default https://$SITE_ADDRESS/ready (SITE_ADDRESS from .env)
#   READY_TIMEOUT    seconds to wait for /ready (default 300), READY_INTERVAL between tries (5)
#   DEPLOY_GIT_SYNC  0 = do not touch the checkout (images only)
#
# The whole script is one function called on the last line. Why: bash reads a script while
# running it, and step 2 may replace this very file with a new version; a function is parsed
# completely before it runs, so the old version finishes consistently.

set -euo pipefail

log() { printf '%s deploy: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# Value of KEY in the .env file (last occurrence, everything after the first "=").
env_get() {
  awk -v key="$1" 'index($0, key "=") == 1 { value = substr($0, length(key) + 2) } END { print value }' .env
}

# Set KEY=VALUE in .env: replace the line or append it. Written through `cat >` so the file
# keeps its inode, owner and mode (600: it holds every secret of the stack).
env_set() {
  local key=$1 value=$2 tmp
  tmp=$(mktemp)
  awk -v key="$key" -v value="$value" '
    index($0, key "=") == 1 { if (!done) print key "=" value; done = 1; next }
    { print }
    END { if (!done) print key "=" value }
  ' .env > "$tmp"
  cat "$tmp" > .env
  rm -f "$tmp"
  log "set $key=$value in .env"
}

compose() {
  docker compose -f docker-compose.yml -f docker-compose.prod.yml "$@"
}

git_sync_enabled() {
  [ "${DEPLOY_GIT_SYNC:-1}" != "0" ] && [ -e .git ]
}

checkout() {
  git_sync_enabled || return 0
  log "checking out $1"
  git checkout --quiet --detach "$1"
}

wait_ready() {
  local url=$1 timeout=${READY_TIMEOUT:-300} interval=${READY_INTERVAL:-5} deadline
  deadline=$((SECONDS + timeout))
  log "waiting up to ${timeout}s for $url"
  while :; do
    if curl --fail --silent --show-error --max-time 5 --output /dev/null "$url"; then
      log "ready: $url"
      return 0
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      log "not ready after ${timeout}s: $url"
      return 1
    fi
    sleep "$interval"
  done
}

# up = pull the images of the version in .env, start it, wait for /ready.
up() {
  compose pull --quiet && compose up -d --remove-orphans && wait_ready "$1"
}

prune_images() {
  local new=$1 prev=$2 prefix ref tag
  prefix=$(env_get IMAGE_PREFIX)
  [ -n "$prefix" ] || return 0
  for name in api airflow mlflow; do
    docker image ls --format '{{.Repository}}:{{.Tag}}' "$prefix-$name" | while read -r ref; do
      tag=${ref##*:}
      case $tag in
        "$new" | "$prev" | latest | "<none>") ;;
        *) log "removing old image $ref"; docker image rm "$ref" > /dev/null || true ;;
      esac
    done
  done
}

main() {
  local tag=${1:-} ref=${2:-} prev_tag prev_ref="" site ready_url

  [ -n "$tag" ] || die "usage: deploy.sh <image-tag> [git-ref]"
  # The tag ends up in .env and in image names: allow only what a Docker tag may contain.
  [[ $tag =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || die "invalid image tag: $tag"
  if [ -z "$ref" ] && [[ $tag =~ ^sha-([0-9a-f]{7,40})$ ]]; then
    ref=${BASH_REMATCH[1]}
  fi
  [ -z "$ref" ] || [[ $ref =~ ^[0-9a-f]{7,40}$ ]] || die "invalid git ref: $ref"

  cd "${APP_DIR:-$(dirname "$0")/../..}"
  [ -f .env ] || die "no .env in $(pwd) (run infra/scripts/bootstrap-server.sh first)"

  # One deploy at a time (the workflow has a concurrency group; this also covers manual runs).
  if command -v flock > /dev/null; then
    exec 9> .deploy.lock
    flock -n 9 || die "another deploy is running"
  fi

  prev_tag=$(env_get IMAGE_TAG)
  site=$(env_get SITE_ADDRESS)
  ready_url=${READY_URL:-https://$site/ready}
  [ -n "${READY_URL:-}" ] || [ -n "$site" ] || die "SITE_ADDRESS is not set in .env"
  log "deploying $tag (running: ${prev_tag:-nothing})"

  if git_sync_enabled; then
    prev_ref=$(git rev-parse HEAD)
    if [ -n "$ref" ]; then
      git fetch --quiet origin
      checkout "$ref"
    fi
  fi

  env_set IMAGE_TAG "$tag"
  if up "$ready_url"; then
    log "deployed $tag"
    prune_images "$tag" "$prev_tag"
    return 0
  fi

  log "deploy of $tag failed; last API log lines:"
  compose logs --tail 30 api >&2 || true
  # `latest` is no rollback target: CI moves it to the newest build, i.e. the failed one.
  if [ -z "$prev_tag" ] || [ "$prev_tag" = "$tag" ] || [ "$prev_tag" = latest ]; then
    die "nothing to roll back to (previous tag: ${prev_tag:-none})"
  fi

  log "rolling back to $prev_tag"
  [ -z "$prev_ref" ] || checkout "$prev_ref"
  env_set IMAGE_TAG "$prev_tag"
  if compose up -d --remove-orphans && wait_ready "$ready_url"; then
    die "rolled back to $prev_tag; $tag was NOT deployed"
  fi
  die "rollback to $prev_tag did not become ready either: check the server by hand"
}

main "$@"
exit
