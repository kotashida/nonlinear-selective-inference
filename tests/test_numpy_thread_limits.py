import ast
from pathlib import Path


THREAD_LIMITS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _numpy_import_lines(tree):
    lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "numpy" or alias.name.startswith("numpy.")
            for alias in node.names
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and (
            node.module == "numpy" or (node.module or "").startswith("numpy.")
        ):
            lines.append(node.lineno)
    return lines


def _thread_limit_assignments(tree):
    assignments = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "os"
            and target.value.attr == "environ"
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
            and isinstance(node.value, ast.Constant)
            and node.value.value == "1"
        ):
            continue
        assignments[target.slice.value] = node.lineno
    return assignments


def test_every_numpy_import_sets_thread_limits_first():
    failures = []
    for directory in ("src", "examples", "tests"):
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            numpy_lines = _numpy_import_lines(tree)
            if not numpy_lines:
                continue
            first_numpy = min(numpy_lines)
            assignments = _thread_limit_assignments(tree)
            missing_or_late = [
                name
                for name in THREAD_LIMITS
                if name not in assignments or assignments[name] >= first_numpy
            ]
            if missing_or_late:
                failures.append(
                    f"{path.relative_to(PROJECT_ROOT)}: "
                    + ", ".join(missing_or_late)
                )

    assert not failures, "Thread limits missing or set after NumPy import:\n" + "\n".join(
        failures
    )
