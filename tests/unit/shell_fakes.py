"""Run the server scripts (infra/scripts/*.sh) against fake `docker`, `curl` and `git`.

The fakes are small bash scripts put first on PATH. They record every call in a log file and
behave according to environment variables, so a test can say "this image tag never becomes
ready" or "pg_dump fails" without Docker, a network or a server:
    FAKE_UNREADY_TAGS     space-separated tags whose /ready never answers 200 (no model yet)
    FAKE_DOWN_TAGS        tags that answer nothing at all (/health, /ready, /recommendations)
    FAKE_BAD_RECS_TAGS    tags whose /recommendations answers 500
    FAKE_SHORT_RECS_TAGS  tags whose /recommendations returns 3 movies instead of 10
    FAKE_EMPTY_RECS_TAGS  tags whose /recommendations returns no movies (empty catalog)
    FAKE_SIGNAL           signal (TERM, HUP, ...) curl sends to its caller on the first /ready
                          check of FAKE_SIGNAL_TAG: "the deploy is killed while it waits"
    FAKE_BAD_PULL_TAGS    space-separated tags whose `docker compose pull` fails
    FAKE_IMAGE_LS_FAIL    1 = `docker image ls` fails
    FAKE_PG_DUMP_FAIL     1 = `pg_dump` fails
    FAKE_IMAGES           file with `repository:tag` lines for `docker image ls`
    FAKE_GIT_HEAD         commit the fake checkout starts at
    FAKE_GIT_ANCESTORS    space-separated "old:new" commit pairs (7-char prefixes): `git
                          merge-base --is-ancestor old new` succeeds for them (and for a == b)
    FAKE_SSHD_FAIL        1 = `sshd -t` rejects the configuration
State (the running tag, the checked-out commit) lives in files under FAKE_STATE.

bash: /bin/bash on Linux (CI); on Windows Git's bash, never System32\\bash.exe (that is WSL,
a different machine with different paths).
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "infra" / "scripts"


def find_bash() -> str | None:
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")
    candidates = []
    if git:
        for parent in Path(git).resolve().parents[:3]:
            candidates += [parent / "usr" / "bin" / "bash.exe", parent / "bin" / "bash.exe"]
    candidates.append(Path(r"C:\Program Files\Git\usr\bin\bash.exe"))
    return next((str(c) for c in candidates if c.is_file()), None)


def posix(path: Path) -> str:
    """Path as bash sees it: C:\\x\\y -> /c/x/y under Git's bash (':' separates PATH)."""
    text = path.as_posix()
    if os.name == "nt" and len(text) > 1 and text[1] == ":":
        return f"/{text[0].lower()}{text[2:]}"
    return text


FAKE_DOCKER = r"""#!/usr/bin/env bash
# fake docker: log the call, act out the few subcommands the scripts use.
printf 'docker %s\n' "$*" >> "$FAKE_LOG"
tag() { awk 'index($0, "IMAGE_TAG=") == 1 { v = substr($0, 11) } END { print v }' .env; }
in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }
if [ "$1" = compose ]; then
  args=" $* "
  case $args in
    *" pull "*)
      if in_list "$(tag)" "${FAKE_BAD_PULL_TAGS:-}"; then echo "pull failed" >&2; exit 1; fi ;;
    *" up "*)
      printf 'up %s\n' "$(tag)" >> "$FAKE_LOG"
      tag > "$FAKE_STATE/running" ;;
    *" logs "*) echo "fake api log line" ;;
    *" pg_dump "*)
      [ "${FAKE_PG_DUMP_FAIL:-0}" = 1 ] && { echo "pg_dump: connection refused" >&2; exit 1; }
      printf 'DUMP %s\n' "${*: -1}" ;;
    *" tar "*) printf 'ARTIFACTS\n' ;;
  esac
  exit 0
fi
if [ "$1 $2" = "image ls" ]; then
  [ "${FAKE_IMAGE_LS_FAIL:-0}" = 1 ] && { echo "image ls failed" >&2; exit 1; }
  repo=${*: -1}
  [ -n "${FAKE_IMAGES:-}" ] && grep "^$repo:" "$FAKE_IMAGES"
  exit 0
fi
exit 0
"""

FAKE_CURL = r"""#!/usr/bin/env bash
# fake curl: answers for the running tag (see the FAKE_* variables in the module docstring).
url=${*: -1}
printf 'curl %s\n' "$url" >> "$FAKE_LOG"
printf 'curl-args %s\n' "$*" >> "$FAKE_LOG"
in_list() { case " $2 " in *" $1 "*) return 0 ;; esac; return 1; }
running=$(cat "$FAKE_STATE/running" 2>/dev/null)
[ -n "$running" ] || exit 7
in_list "$running" "${FAKE_DOWN_TAGS:-}" && exit 7
case $url in
  */ready)
    if [ -n "${FAKE_SIGNAL:-}" ] && [ "$running" = "${FAKE_SIGNAL_TAG:-}" ] \
        && [ ! -e "$FAKE_STATE/signalled" ]; then
      touch "$FAKE_STATE/signalled"
      printf 'signal %s\n' "$FAKE_SIGNAL" >> "$FAKE_LOG"
      kill -s "$FAKE_SIGNAL" "$PPID"
      exit 22
    fi
    in_list "$running" "${FAKE_UNREADY_TAGS:-}" && exit 22 ;;
  */recommendations*)
    in_list "$running" "${FAKE_BAD_RECS_TAGS:-}" && exit 22
    n=10
    in_list "$running" "${FAKE_SHORT_RECS_TAGS:-}" && n=3
    in_list "$running" "${FAKE_EMPTY_RECS_TAGS:-}" && n=0
    items=""
    for ((i = 1; i <= n; i++)); do
      items+="${items:+,}{\"movie_id\":$i,\"title\":\"Movie $i\",\"year\":2000,\"genres\":[]}"
    done
    printf '{"items":[%s],"path":"cold_start","model_version":null,"n_ratings":0}' "$items" ;;
esac
exit 0
"""

FAKE_GIT = r"""#!/usr/bin/env bash
# fake git: remembers the checked-out commit.
printf 'git %s\n' "$*" >> "$FAKE_LOG"
head_file="$FAKE_STATE/head"
[ -f "$head_file" ] || printf '%s\n' "${FAKE_GIT_HEAD:-0000000}" > "$head_file"
case $1 in
  rev-parse) cat "$head_file" ;;
  checkout) printf '%s\n' "${*: -1}" > "$head_file" ;;
  merge-base)  # merge-base --is-ancestor <old> <new>
    old=${3:0:7} new=${4:0:7}
    [ "$old" = "$new" ] && exit 0
    case " ${FAKE_GIT_ANCESTORS:-} " in *" $old:$new "*) exit 0 ;; esac
    exit 1 ;;
esac
exit 0
"""

FAKE_SSHD = r"""#!/usr/bin/env bash
printf 'sshd %s\n' "$*" >> "$FAKE_LOG"
[ "${FAKE_SSHD_FAIL:-0}" = 1 ] && { echo "sshd: bad configuration" >&2; exit 255; }
exit 0
"""

# Commands that only log their arguments (system tools the bootstrap script calls).
FAKE_LOGGER = r"""#!/usr/bin/env bash
printf '%s %s\n' "$(basename "$0")" "$*" >> "$FAKE_LOG"
exit 0
"""


@dataclass
class Result:
    returncode: int
    output: str
    calls: list[str]
    state: Path

    def file(self, name: str) -> str:
        path = self.state / name
        return path.read_text().strip() if path.exists() else ""


class ShellHarness:
    """A fake bin directory + state directory under `tmp`, and a way to run a script."""

    def __init__(self, tmp: Path, bash: str) -> None:
        self.bash = bash
        self.bin = tmp / "fake-bin"
        self.state = tmp / "fake-state"
        self.log = tmp / "calls.log"
        self.bin.mkdir()
        self.state.mkdir()
        fakes = {"docker": FAKE_DOCKER, "curl": FAKE_CURL, "git": FAKE_GIT, "sshd": FAKE_SSHD}
        fakes |= {name: FAKE_LOGGER for name in ("systemctl", "ufw")}
        for name, body in fakes.items():
            path = self.bin / name
            path.write_text(body, newline="\n")
            path.chmod(0o755)

    def run(self, script: str, *args: str, cwd: Path, **env: str) -> Result:
        return self._run([posix(SCRIPTS / script), *args], cwd, env)

    def call(self, script: str, function: str, *args: str, cwd: Path, **env: str) -> Result:
        """Source the script (defines its functions, runs no main) and call one function."""
        code = f'source "{posix(SCRIPTS / script)}" && {function} "$@"'
        return self._run(["-c", code, "bash", *args], cwd, env)

    def _run(self, bash_args: list[str], cwd: Path, env: dict[str, str]) -> Result:
        environ = {
            **os.environ,
            "FAKE_BIN": posix(self.bin),
            "FAKE_LOG": posix(self.log),
            "FAKE_STATE": posix(self.state),
            "APP_DIR": posix(cwd),
            **env,
        }
        # PATH is prepended inside bash, after Git's bash has set up its own PATH.
        command = [self.bash, "-c", 'PATH="$FAKE_BIN:$PATH"; exec bash "$@"', "bash", *bash_args]
        proc = subprocess.run(
            command,
            env=environ,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        calls = self.log.read_text().splitlines() if self.log.exists() else []
        return Result(proc.returncode, proc.stdout + proc.stderr, calls, self.state)
