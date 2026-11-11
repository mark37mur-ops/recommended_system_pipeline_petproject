#!/usr/bin/env bash
# Nightly backup of what cannot be rebuilt (spec 6.2), one dated archive per run:
#   app.dump            pg_dump of `app`: movies, users, and the web ratings, the only data
#                       that exists nowhere else (MovieLens itself can be downloaded again)
#   mlflow.dump         pg_dump of `mlflow`: runs, metrics, the registry and the champion alias
#   mlartifacts.tar.gz  the MLflow artifact store (the model files the registry points to);
#                       without it a restored registry would point at nothing
# into $BACKUP_DIR/backup-<UTC timestamp>.tar, keeping the newest $BACKUP_KEEP archives.
#
#   infra/scripts/backup.sh                       # by hand, from the checkout on the server
#   30 3 * * * deploy /opt/movielens-recsys/infra/scripts/backup.sh   # cron (bootstrap sets it)
#
# Environment (optional): APP_DIR (default: two levels above this script),
#   BACKUP_DIR (default $APP_DIR/backups), BACKUP_KEEP (default 7).
#
# Why this way:
#   - pg_dump runs inside the postgres container (`docker compose exec`): no published
#     database port, no password (local socket); -Fc = compressed custom format, restorable
#     table by table with pg_restore;
#   - the artifacts are read by the mlflow container itself (its image has tar), no extra image;
#   - everything is written to a temporary directory first and renamed into place at the end:
#     an interrupted run never leaves a half-written file that looks like a backup;
#   - umask 077: the archive holds user data, only the deploy user can read it.
#   It stays on the same disk as the data, a known compromise (spec 6.2); copying it off the
#   server (rclone, object storage) is v2.
#
# Restore (stack running, from the checkout):
#   tar -xf backups/backup-<ts>.tar -C /tmp/restore
#   docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T postgres \
#     pg_restore -U postgres -d app --clean --if-exists < /tmp/restore/app.dump
#   (same for mlflow; artifacts: ... exec -T mlflow tar -C /mlartifacts -xzf - < mlartifacts.tar.gz)

set -euo pipefail

log() { printf '%s backup: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2; }

compose() {
  docker compose -f docker-compose.yml -f docker-compose.prod.yml "$@"
}

main() {
  local backup_dir keep stamp archive old
  cd "${APP_DIR:-$(dirname "$0")/../..}"
  backup_dir=${BACKUP_DIR:-$(pwd)/backups}
  keep=${BACKUP_KEEP:-7}
  [[ $keep =~ ^[1-9][0-9]*$ ]] || { log "BACKUP_KEEP must be a positive integer"; exit 1; }

  umask 077
  mkdir -p "$backup_dir"
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  archive="$backup_dir/backup-$stamp.tar"
  # Global, not `local`: the EXIT trap runs after main has returned and must still see it.
  work=$(mktemp -d "$backup_dir/.tmp-$stamp.XXXXXX")
  # Whatever happens below (a failed dump included), the temporary directory goes away.
  trap 'rm -rf "$work"' EXIT

  for db in app mlflow; do
    log "pg_dump $db"
    compose exec -T postgres pg_dump -U postgres -Fc "$db" > "$work/$db.dump"
  done
  log "MLflow artifacts"
  compose exec -T mlflow tar -C /mlartifacts -czf - . > "$work/mlartifacts.tar.gz"

  # Relative names inside the work directory (GNU tar would read "C:/..." as host:path).
  (cd "$work" && tar -cf archive.tar app.dump mlflow.dump mlartifacts.tar.gz)
  mv "$work/archive.tar" "$archive"
  log "written $archive ($(du -h "$archive" | cut -f1))"

  # Retention: names sort by time (UTC timestamp), so the newest come first in reverse order.
  find "$backup_dir" -maxdepth 1 -name 'backup-*.tar' | sort -r | tail -n "+$((keep + 1))" \
    | while read -r old; do
        log "removing old $old"
        rm -f "$old"
      done
}

main "$@"
