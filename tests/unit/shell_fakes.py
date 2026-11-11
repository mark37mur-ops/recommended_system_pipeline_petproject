"""Run the server scripts (infra/scripts/*.sh) against fake `docker`, `curl` and `git`.

The fakes are small bash scripts put first on PATH. They record every call in a log file and
behave according to environment variables, so a test can say "this image tag never becomes
ready" or "pg_dump fails" without Docker, a network or a server:
    FAKE_UNREADY_TAGS   space-separated tags whose /ready never answers 200
    FAKE_BAD_PULL_TAGS  space-separated tags whose `docker compose pull` fails
    FAKE_PG_DUMP_FAIL   1 = `pg_dump` fails
    FAKE_IMAGES         file with `repository:tag` lines for `docker image ls`
    FAKE_GIT_HEAD       commit the fake checkout starts at
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
  repo=${*: -1}
  [ -n "${FAKE_IMAGES:-}" ] && grep "^$repo:" "$FAKE_IMAGES"
  exit 0
fi
exit 0
"""

FAKE_CURL = r"""#!/usr/bin/env bash
# fake curl: /ready answers unless the running tag is listed in FAKE_UNREADY_TAGS.
printf 'curl %s\n' "${*: -1}" >> "$FAKE_LOG"
running=$(cat "$FAKE_STATE/running" 2>/dev/null)
case " ${FAKE_UNREADY_TAGS:-} " in *" $running "*) exit 22 ;; esac
[ -n "$running" ] || exit 7
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
esac
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
        for name, body in {"docker": FAKE_DOCKER, "curl": FAKE_CURL, "git": FAKE_GIT}.items():
            path = self.bin / name
            path.write_text(body, newline="\n")
            path.chmod(0o755)

    def run(self, script: str, *args: str, cwd: Path, **env: str) -> Result:
        environ = {
            **os.environ,
            "FAKE_BIN": posix(self.bin),
            "FAKE_LOG": posix(self.log),
            "FAKE_STATE": posix(self.state),
            "APP_DIR": posix(cwd),
            **env,
        }
        # PATH is prepended inside bash, after Git's bash has set up its own PATH.
        command = [self.bash, "-c", 'PATH="$FAKE_BIN:$PATH"; exec bash "$@"', "bash"]
        command += [posix(SCRIPTS / script), *args]
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
