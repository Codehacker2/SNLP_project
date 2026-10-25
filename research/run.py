"""Run research stages in one shared session, with outputs isolated from the app.

No arguments lists stages without executing them. Research dependencies, datasets
and trained models are needed only when you explicitly select --through.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

RESEARCH = Path(__file__).resolve().parent
RUNS = RESEARCH / "runs"
MANIFEST = RESEARCH / "source_map.json"


def _write_notebook_file(target: str, resource: str) -> None:
    """Replace %%writefile using its preserved UTF-8 payload in the current run."""
    body = (RESEARCH / resource).read_text(encoding="utf-8")
    # Notebook execution normalizes a cell's trailing newline before %%writefile.
    if not body.endswith("\n"):
        body += "\n"
    Path(target).write_text(body, encoding="utf-8")
    print(f"Wrote {target}")


def run_stages(stages: list[dict], namespace: dict) -> None:
    for stage in stages:
        path = RESEARCH / stage["file"]
        print(f"\nRunning {stage['name']}: {stage['description']}", flush=True)
        namespace["__file__"] = str(path)
        try:
            # Do not inherit this runner's future-annotations compiler setting.
            code = compile(path.read_text(encoding="utf-8"), str(path), "exec", dont_inherit=True)
            exec(code, namespace)
        except BaseException:
            print(f"\nFailed in {stage['name']}. Match the traceback to the '# %%' "
                  "cell marker or research/source_map.json.", file=sys.stderr)
            raise


def create_run_directory(name: str | None = None) -> Path:
    """A run must be new/empty and contained within research/runs/."""
    runs_root = RUNS.resolve()
    if runs_root != RESEARCH / "runs":
        raise ValueError("research/runs must not redirect outside the research directory")
    name = name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = (runs_root / name).resolve()
    if path == runs_root or not path.is_relative_to(runs_root):
        raise ValueError("Run directory must be a child of research/runs/")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Use a new or empty run directory; previous outputs are preserved")
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_research(stages: list[dict], run_dir: Path, *, display) -> None:
    previous_cwd = Path.cwd()
    previous_env = {key: os.environ.get(key) for key in ("G1_WORKDIR", "G1_REPO")}
    previous_path = sys.path[:]
    try:
        # Override inherited notebook settings to protect the application/data.
        os.environ["G1_WORKDIR"] = str(run_dir / "work")
        os.environ["G1_REPO"] = str(run_dir / "repository")
        os.chdir(run_dir)
        print(f"Research outputs: {run_dir}", flush=True)
        namespace = {"__name__": "__main__", "__doc__": None, "__package__": None,
                     "display": display, "_write_notebook_file": _write_notebook_file}
        run_stages(stages, namespace)
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main(argv=None) -> int:
    stages = json.loads(MANIFEST.read_text(encoding="utf-8"))["stages"]
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="list stages without execution (default)")
    action.add_argument("--through", choices=[s["name"] for s in stages] + ["all"],
                        help="run from setup through this stage in notebook order")
    parser.add_argument("--run-dir", help="new run folder name under research/runs/")
    args = parser.parse_args(argv)
    if args.through is None:
        for stage in stages:
            print(f"{stage['name']:36s} {stage['description']}")
        return 0
    try:
        from IPython.display import display
    except ImportError:
        parser.error("Research execution requires IPython; install research/requirements.txt")
    try:
        run_dir = create_run_directory(args.run_dir)
    except ValueError as exc:
        parser.error(str(exc))
    last = len(stages) - 1 if args.through == "all" else next(
        index for index, stage in enumerate(stages) if stage["name"] == args.through
    )
    run_research(stages[:last + 1], run_dir, display=display)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
