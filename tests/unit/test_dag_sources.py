"""Static checks of the DAG files that need no Airflow: every recsys function a task calls
exists, and the branch ids returned by decide_promotion are task ids of the DAG.

(The full DagBag test lives in tests/dags and runs inside the Airflow image.)
"""

import ast
import importlib
from pathlib import Path

import pytest

from recsys.pipelines.retrain import PROMOTE_TASK, SKIP_TASK

DAGS = Path(__file__).resolve().parents[2] / "airflow" / "dags"
DAG_FILES = sorted(DAGS.glob("*.py"))


def _recsys_calls(tree: ast.AST) -> set[tuple[str, str]]:
    """(module, attribute) for `from recsys.pipelines import m` + `m.attr(...)` in bodies."""
    modules: dict[str, str] = {}
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("recsys"):
            for alias in node.names:
                modules[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in modules
        ):
            calls.add((modules[node.func.value.id], node.func.attr))
    return calls


def test_there_are_two_dag_files():
    assert [p.name for p in DAG_FILES] == ["bootstrap_data.py", "retrain_daily.py"]


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
def test_dag_files_import_recsys_only_inside_task_bodies(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:  # module level
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = (
                [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            )
            assert not any((n or "").startswith("recsys") for n in names), ast.dump(node)


@pytest.mark.parametrize("path", DAG_FILES, ids=lambda p: p.name)
def test_every_called_recsys_function_exists(path):
    calls = _recsys_calls(ast.parse(path.read_text(encoding="utf-8")))
    assert calls
    for module, attr in calls:
        assert callable(getattr(importlib.import_module(module), attr, None)), (module, attr)


def test_branch_ids_are_task_ids_of_retrain_daily():
    source = (DAGS / "retrain_daily.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert PROMOTE_TASK in functions  # @task def promote -> task id "promote"
    assert f'task_id="{SKIP_TASK}"' in source  # EmptyOperator(task_id="skip")
