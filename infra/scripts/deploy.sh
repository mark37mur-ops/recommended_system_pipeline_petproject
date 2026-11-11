#!/usr/bin/env bash
# Deploy one version of the stack on the server, roll back if it does not come up (spec 6.4).
#
#   infra/scripts/deploy.sh [--allow-older] <image-tag> [git-ref]
#   infra/scripts/deploy.sh sha-1a2b3c4                  # git ref defaults to 1a2b3c4
#   infra/scripts/deploy.sh --detach <id> sha-1a2b3c4 1a2b3c4d5e6f...
#                                                        # what .github/workflows/deploy.yml runs
#
# Steps:
#   1. one deploy at a time: wait for the lock (.deploy.lock, DEPLOY_LOCK_TIMEOUT seconds);
#   2. remember the running version (IMAGE_TAG in .env, the checked-out commit) and how
#      healthy it is: that picks the smoke gate (below);
#   3. refuse a commit that is OLDER than the running one (an ancestor of it), exit 3: a late
#      or re-run CI must not move the server back in history. --allow-older = a human asked
#      for exactly that (the workflow's manual run, i.e. a rollback);
#   4. check out the commit being deployed: docker-compose*.yml and the configs under infra/
#      (Caddyfile, Grafana provisioning, prometheus.yml) are read from this directory, so they
#      must match the code inside the images;
#   5. write the new IMAGE_TAG into .env, `docker compose pull`, `up -d --remove-orphans`;
#   6. smoke test through Caddy (spec 6.5), up to READY_TIMEOUT seconds. Not
#      `docker compose up --wait`: it fails on the one-shot jobs that exit by design;
#   7. failed pull, failed up, failed smoke test, or the script is told to stop (SIGTERM,
#      SIGINT, SIGHUP) -> put the previous tag and commit back, `up -d` again, smoke-test
#      again, exit 1;
#   8. success -> delete images of this project other than the new, the previous and `latest`
#      (every version is up to ~1 GB of new layers; the previous one stays for a fast rollback).
#
# Smoke gates (chosen in step 2, SMOKE_GATE=full|serving overrides):
#   full     the running version answers /ready 200 (model loaded, database up, catalog
#            loaded): the new one must too, and GET /recommendations?k=10 without a cookie
#            (the cold-start path, so no visitor is created and nothing is written) must
#            return 10 movies. The personalised path is covered by /ready itself: the API
#            loads a model only after a smoke prediction with it (`validate` in main.py), so
#            a new code version that cannot run the champion never becomes ready;
#   serving  nothing ready runs (the first deploy; or no champion yet: bootstrap_data and
#            retrain_daily can only run once the stack is up, and until then the API serves
#            popularity with /ready 503 by design, spec 3.5): /health must answer 200 and
#            /recommendations must answer a JSON list (empty while the catalog is not
#            loaded). A stricter gate would fail every deploy until a first model exists.
#
# --detach <id>: check the arguments, start the deploy in its own session (setsid) with all
# output in .deploy/<id>.log and return at once; .deploy/<id>.status says `running`, then
# `done <exit code>`. The workflow polls that file. Why: a deploy run inside the SSH session
# died with it (runner timeout, cancel, a network blip) between `up` and the rollback.
#
# Environment (all optional):
#   APP_DIR              checkout with .env (default: two levels above this script)
#   SMOKE_URL            default https://$SITE_ADDRESS, reached at 127.0.0.1 (curl --resolve:
#                        through Caddy and its certificate, without depending on DNS)
#   READY_TIMEOUT        seconds for the smoke test (default 300), READY_INTERVAL between tries (5)
#   SMOKE_GATE           auto (default), full or serving
#   DEPLOY_LOCK_TIMEOUT  seconds to wait for a running deploy (default 900)
#   DEPLOY_GIT_SYNC      0 = do not touch the checkout (images only)
#
# The whole script is one function called on the last line. Why: bash reads a script while
# running it, and step 4 may replace this very file with a new version; a function is parsed
# completely before it runs, so the old version finishes consistently.

set -euo pipefail

EXIT_OLDER=3
SMOKE_K=10

# A closed reader (the SSH channel of a cancelled job) must not kill the script on the next
# line of output: with SIGPIPE ignored the write just fails, and log() ignores that.
log() { printf '%s deploy: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2 || true; }
die() { log "ERROR: $*"; exit 1; }

# Value of KEY in the .env file (last occurrence, everything after the first "=").
env_get() {
  awk -v key="$1" 'index($0, key "=") == 1 { value = substr($0, length(key) + 2) } END { print value }' .env
}

# Set KEY=VALUE in .env: replace the line or append it. Written to a temporary file next to
# .env and renamed over it: a rename is atomic, so an interrupted or failed write (full disk)
# never leaves a truncated .env, which holds every secret of the stack. mktemp creates the
# file with mode 600, the mode .env has; the owner is the deploy user, as before.
env_set() {
  local key=$1 value=$2 tmp
  tmp=$(mktemp .env.XXXXXX)
  if ! awk -v key="$key" -v value="$value" '
      index($0, key "=") == 1 { if (!done) print key "=" value; done = 1; next }
      { print }
      END { if (!done) print key "=" value }
    ' .env > "$tmp" || ! mv -f "$tmp" .env; then
    rm -f "$tmp"
    die "cannot write .env"
  fi
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

# ----- smoke test ---------------------------------------------------------------------------
CURL_OPTS=()
SMOKE_BASE=""

http_ok() {
  curl --fail --silent --show-error --max-time 5 --output /dev/null "${CURL_OPTS[@]}" "$1"
}

wait_for() {
  local url=$1 timeout=${READY_TIMEOUT:-300} interval=${READY_INTERVAL:-5} deadline
  deadline=$((SECONDS + timeout))
  log "waiting up to ${timeout}s for $url"
  while :; do
    if http_ok "$url"; then
      log "ok: $url"
      return 0
    fi
    if [ "$SECONDS" -ge "$deadline" ]; then
      log "no answer after ${timeout}s: $url"
      return 1
    fi
    sleep "$interval"
  done
}

# recommendations_ok <gate>: a visitor without a cookie gets the cold-start list.
recommendations_ok() {
  local gate=$1 url="$SMOKE_BASE/recommendations?k=$SMOKE_K" body n attempt
  for attempt in 1 2 3; do
    if body=$(curl --fail --silent --show-error --max-time 10 "${CURL_OPTS[@]}" "$url"); then
      n=$(printf '%s' "$body" | { grep -o '"movie_id":' || true; } | wc -l)
      n=$((n))
      case $body in
        *'"items":['*)
          if [ "$gate" = serving ] || [ "$n" -eq "$SMOKE_K" ]; then
            log "ok: $url returned $n movies"
            return 0
          fi
          log "$url returned $n movies, expected $SMOKE_K (try $attempt)" ;;
        *) log "$url answered without an items list (try $attempt)" ;;
      esac
    fi
    sleep "${READY_INTERVAL:-5}"
  done
  return 1
}

smoke() {
  case $1 in
    full) wait_for "$SMOKE_BASE/ready" && recommendations_ok full ;;
    serving) wait_for "$SMOKE_BASE/health" && recommendations_ok serving ;;
    *) log "unknown smoke gate: $1"; return 1 ;;
  esac
}

choose_gate() {
  case ${SMOKE_GATE:-auto} in
    full | serving) echo "$SMOKE_GATE" ;;
    *) if http_ok "$SMOKE_BASE/ready"; then echo full; else echo serving; fi ;;
  esac
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

# ----- one deploy at a time -----------------------------------------------------------------
lock() {
  if ! command -v flock > /dev/null; then
    log "WARNING: flock not found, deploying without a lock"
    return 0
  fi
  exec 9> .deploy.lock
  flock -n 9 && return 0
  log "another deploy is running: waiting up to ${DEPLOY_LOCK_TIMEOUT:-900}s for it"
  flock -w "${DEPLOY_LOCK_TIMEOUT:-900}" 9 || die "another deploy is still running"
}

# ----- rollback (also from a signal) --------------------------------------------------------
# Globals, not locals of main: the signal handler must see them wherever the signal lands.
TAG="" PREV_TAG="" PREV_REF="" GATE="" SWITCHED=0

rollback() {
  trap '' TERM INT HUP  # a signal must not cut the rollback short
  log "deploy of $TAG failed ($1); last API log lines:"
  compose logs --tail 30 api >&2 || true
  # `latest` is no rollback target: CI moves it to the newest build, i.e. the failed one.
  if [ -z "$PREV_TAG" ] || [ "$PREV_TAG" = "$TAG" ] || [ "$PREV_TAG" = latest ]; then
    die "nothing to roll back to (previous tag: ${PREV_TAG:-none})"
  fi
  log "rolling back to $PREV_TAG"
  [ -z "$PREV_REF" ] || checkout "$PREV_REF"
  env_set IMAGE_TAG "$PREV_TAG"
  if compose up -d --remove-orphans && smoke "$GATE"; then
    die "rolled back to $PREV_TAG; $TAG was NOT deployed"
  fi
  die "rollback to $PREV_TAG did not pass the smoke test either: check the server by hand"
}

# shellcheck disable=SC2329  # called from the traps in main
on_signal() {
  trap '' TERM INT HUP
  log "interrupted by SIG$1"
  [ "$SWITCHED" = 1 ] || die "interrupted by SIG$1 before anything was changed"
  rollback "interrupted by SIG$1"
}

# ----- detached run -------------------------------------------------------------------------
# start_detached <id> <script> <arguments of the real run...>
start_detached() {
  local id=$1 script=$2 dir starter=()
  shift 2
  dir=$PWD/.deploy
  mkdir -p "$dir"
  # A command sent twice (SSH retry) must not start a second deploy.
  [ ! -e "$dir/$id.status" ] || die "deploy $id was already started (see $dir/$id.log)"
  find "$dir" -type f -mtime +30 -delete 2> /dev/null || true
  printf 'running\n' > "$dir/$id.status"
  # setsid: own session, so the end of the SSH session (SIGHUP to its process group) does
  # not reach it; stdin/stdout/stderr point at files, not at the SSH channel.
  command -v setsid > /dev/null && starter=(setsid)
  DEPLOY_STATUS_FILE="$dir/$id.status" APP_DIR=$PWD \
    "${starter[@]}" nohup bash "$script" "$@" < /dev/null >> "$dir/$id.log" 2>&1 &
  log "started deploy $id in the background: status $dir/$id.status, log $dir/$id.log"
}

# shellcheck disable=SC2329  # called from the EXIT trap in main
write_status() {
  printf 'done %s\n' "$1" > "$DEPLOY_STATUS_FILE.tmp" && mv -f "$DEPLOY_STATUS_FILE.tmp" "$DEPLOY_STATUS_FILE"
}

main() {
  local detach="" allow_older=0 ref site script
  # First thing, before any check: a detached child that dies in validation (a bad tag, no
  # .env) must leave `done <code>` for the workflow, not `running` forever.
  [ -z "${DEPLOY_STATUS_FILE:-}" ] || trap 'write_status $?' EXIT
  script=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
  while [ $# -gt 0 ]; do
    case $1 in
      --detach) [ $# -ge 2 ] || die "--detach needs an id"; detach=$2; shift 2 ;;
      --allow-older) allow_older=1; shift ;;
      *) break ;;
    esac
  done
  TAG=${1:-}
  ref=${2:-}

  [ -n "$TAG" ] || die "usage: deploy.sh [--detach <id>] [--allow-older] <image-tag> [git-ref]"
  # The tag ends up in .env and in image names: allow only what a Docker tag may contain.
  [[ $TAG =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || die "invalid image tag: $TAG"
  if [ -z "$ref" ] && [[ $TAG =~ ^sha-([0-9a-f]{7,40})$ ]]; then
    ref=${BASH_REMATCH[1]}
  fi
  [ -z "$ref" ] || [[ $ref =~ ^[0-9a-f]{7,40}$ ]] || die "invalid git ref: $ref"
  [ -z "$detach" ] || [[ $detach =~ ^[A-Za-z0-9_-]{1,64}$ ]] || die "invalid run id: $detach"
  case ${SMOKE_GATE:-auto} in
    auto | full | serving) ;;
    *) die "SMOKE_GATE must be auto, full or serving" ;;
  esac

  cd "${APP_DIR:-$(dirname "$0")/../..}"
  [ -f .env ] || die "no .env in $(pwd) (run infra/scripts/bootstrap-server.sh first)"

  if [ -n "$detach" ]; then
    local args=()
    [ "$allow_older" = 0 ] || args+=(--allow-older)
    start_detached "$detach" "$script" "${args[@]}" "$TAG" ${ref:+"$ref"}
    return 0
  fi

  trap '' PIPE
  trap 'on_signal TERM' TERM
  trap 'on_signal INT' INT
  trap 'on_signal HUP' HUP

  lock

  PREV_TAG=$(env_get IMAGE_TAG)
  site=$(env_get SITE_ADDRESS)
  if [ -n "${SMOKE_URL:-}" ]; then
    SMOKE_BASE=${SMOKE_URL%/}
  else
    [ -n "$site" ] || die "SITE_ADDRESS is not set in .env"
    SMOKE_BASE=https://$site
    CURL_OPTS=(--resolve "$site:443:127.0.0.1")
  fi
  log "deploying $TAG (running: ${PREV_TAG:-nothing})"
  GATE=$(choose_gate)
  log "smoke gate: $GATE"

  if git_sync_enabled; then
    PREV_REF=$(git rev-parse HEAD)
    if [ -n "$ref" ]; then
      git fetch --quiet origin
      git rev-parse --verify --quiet "$ref^{commit}" > /dev/null || die "unknown commit: $ref"
      if [ "$allow_older" = 0 ] && git merge-base --is-ancestor "$ref" "$PREV_REF" \
          && ! git merge-base --is-ancestor "$PREV_REF" "$ref"; then
        log "skipped: $ref is older than the running commit $PREV_REF (--allow-older to roll back)"
        exit "$EXIT_OLDER"
      fi
    fi
  fi

  SWITCHED=1
  [ -z "$ref" ] || checkout "$ref"
  env_set IMAGE_TAG "$TAG"
  if compose pull --quiet && compose up -d --remove-orphans && smoke "$GATE"; then
    SWITCHED=0
    log "deployed $TAG"
    prune_images "$TAG" "$PREV_TAG" || log "image prune failed (ignored)"
    return 0
  fi
  rollback "pull, up or smoke test failed"
}

main "$@"
exit
