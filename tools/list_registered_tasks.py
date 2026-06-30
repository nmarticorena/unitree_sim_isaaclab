#!/usr/bin/env python3
"""List Gym task IDs registered by this repository."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegisteredTask:
    """A task ID found in a gym.register(...) call."""

    task_id: str
    source: Path
    line: int


def _literal_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_register_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "register":
        return False
    return isinstance(func.value, ast.Name) and func.value.id in {"gym", "gymnasium"}


def _task_id_from_call(node: ast.Call) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == "id":
            return _literal_string(keyword.value)
    if node.args:
        return _literal_string(node.args[0])
    return None


def discover_registered_tasks(root: Path, prefix: str) -> list[RegisteredTask]:
    tasks: list[RegisteredTask] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as exc:
            raise SystemExit(f"Cannot parse {path}: {exc}") from exc

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_register_call(node):
                continue
            task_id = _task_id_from_call(node)
            if task_id is None or (prefix and not task_id.startswith(prefix)):
                continue
            tasks.append(RegisteredTask(task_id=task_id, source=path, line=node.lineno))

    return sorted(tasks, key=lambda task: (task_group(task), task.task_id))


def task_group(task: RegisteredTask) -> str:
    parts = task.source.parts
    if "g1_tasks" in parts:
        return "G1 tasks"
    if "h1-2_tasks" in parts:
        return "H1-2 tasks"
    return "Other tasks"


def control_mode(task_id: str) -> str:
    if task_id.endswith("-Wholebody"):
        return "Wholebody"
    if task_id.endswith("-Joint"):
        return "Joint"
    return "-"


def source_module(path: Path, repo_root: Path) -> str:
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        relative = path
    if relative.name == "__init__.py":
        return str(relative.parent)
    return str(relative)


def print_table(tasks: list[RegisteredTask], repo_root: Path) -> None:
    print(f"Registered Isaac tasks ({len(tasks)})")
    if not tasks:
        return

    grouped: dict[str, list[RegisteredTask]] = defaultdict(list)
    for task in tasks:
        grouped[task_group(task)].append(task)

    group_order = ["G1 tasks", "H1-2 tasks", "Other tasks"]
    for group in [name for name in group_order if name in grouped]:
        group_tasks = grouped[group]
        id_width = max(len("Task ID"), *(len(task.task_id) for task in group_tasks))
        mode_width = max(len("Mode"), *(len(control_mode(task.task_id)) for task in group_tasks))
        module_width = max(len("Module"), *(len(source_module(task.source, repo_root)) for task in group_tasks))

        print()
        print(f"{group} ({len(group_tasks)})")
        print(f"{'#':>2}  {'Task ID':<{id_width}}  {'Mode':<{mode_width}}  {'Module':<{module_width}}")
        print(f"{'--':>2}  {'-' * id_width}  {'-' * mode_width}  {'-' * module_width}")
        for index, task in enumerate(group_tasks, start=1):
            print(
                f"{index:>2}  "
                f"{task.task_id:<{id_width}}  "
                f"{control_mode(task.task_id):<{mode_width}}  "
                f"{source_module(task.source, repo_root):<{module_width}}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List Gym task IDs registered in this repository.")
    parser.add_argument(
        "roots",
        nargs="*",
        default=["tasks"],
        help="Directories to scan. Defaults to the local tasks directory.",
    )
    parser.add_argument(
        "--prefix",
        default="Isaac-",
        help="Only include task IDs with this prefix. Use an empty value to include all IDs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    tasks: list[RegisteredTask] = []
    for root_arg in args.roots:
        root = (repo_root / root_arg).resolve()
        if not root.exists():
            raise SystemExit(f"Scan root does not exist: {root}")
        tasks.extend(discover_registered_tasks(root, args.prefix))

    print_table(sorted(tasks, key=lambda task: (task_group(task), task.task_id)), repo_root)


if __name__ == "__main__":
    main()
