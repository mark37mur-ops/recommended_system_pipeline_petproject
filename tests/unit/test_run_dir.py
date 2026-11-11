import json

import pytest

from recsys.pipelines.run_dir import (
    RunDir,
    atomic_dir,
    read_json,
    resolve_runs_root,
    safe_run_id,
    write_json_atomic,
)


@pytest.mark.parametrize(
    ("run_id", "expected"),
    [
        ("manual__2026-11-11T10:00:00+00:00", "manual__2026-11-11T10-00-00-00-00"),
        (
            "scheduled__2026-11-10T00:00:00.123456+00:00",
            "scheduled__2026-11-10T00-00-00.123456-00-00",
        ),
        ("a/b\\c d", "a-b-c-d"),
        ("plain-id_1", "plain-id_1"),
    ],
)
def test_safe_run_id_keeps_only_path_safe_characters(run_id, expected):
    assert safe_run_id(run_id) == expected


@pytest.mark.parametrize("run_id", ["", ".", "..", "///"])
def test_safe_run_id_rejects_ids_that_are_not_a_folder_name(run_id):
    with pytest.raises(ValueError):
        safe_run_id(run_id)


def test_run_dir_for_run_lives_under_the_runs_root(tmp_path):
    rd = RunDir.for_run("manual__x:1", runs_root=tmp_path)
    assert rd.root == tmp_path / "manual__x-1"
    assert rd.ratings_path.parent == rd.extract_dir
    assert rd.root in rd.evaluate_json.parents


def test_runs_root_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    assert resolve_runs_root() == tmp_path / "runs"
    assert resolve_runs_root(tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.delenv("RUNS_DIR")
    assert resolve_runs_root().as_posix() == "/data/runs"


def test_atomic_dir_publishes_only_on_success(tmp_path):
    target = tmp_path / "out"
    with pytest.raises(RuntimeError), atomic_dir(target) as tmp:
        (tmp / "half.txt").write_text("partial")
        raise RuntimeError("boom")
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []  # no temp leftovers

    with atomic_dir(target) as tmp:
        (tmp / "a.txt").write_text("first")
    assert (target / "a.txt").read_text() == "first"


def test_atomic_dir_replaces_previous_output_completely(tmp_path):
    target = tmp_path / "out"
    with atomic_dir(target) as tmp:
        (tmp / "old.txt").write_text("old")
    with atomic_dir(target) as tmp:
        (tmp / "new.txt").write_text("new")
    assert sorted(p.name for p in target.iterdir()) == ["new.txt"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_json_roundtrip_is_atomic(tmp_path):
    path = tmp_path / "sub" / "x.json"
    write_json_atomic(path, {"b": 1, "a": [1, 2]})
    assert read_json(path) == {"b": 1, "a": [1, 2]}
    assert json.loads(path.read_text(encoding="utf-8")) == {"b": 1, "a": [1, 2]}
    assert sorted(p.name for p in path.parent.iterdir()) == ["x.json"]
