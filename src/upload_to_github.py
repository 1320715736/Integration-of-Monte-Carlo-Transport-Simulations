#!/usr/bin/env python3
"""One-command GitHub upload helper for this project.

Usage:
  python src/upload_to_github.py
  python src/upload_to_github.py -m "update paper figures"

The script stages the whole repository, creates a commit if there are changes,
and pushes the current branch to the configured remote.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit("This script must be run inside a Git repository.")
    return Path(result.stdout.strip())


def current_branch(root: Path) -> str:
    result = run(["git", "branch", "--show-current"], root)
    branch = result.stdout.strip()
    if not branch:
        raise SystemExit("Cannot determine current Git branch.")
    return branch


def has_changes(root: Path) -> bool:
    result = run(["git", "status", "--porcelain"], root)
    return bool(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage, commit, and push the whole project to GitHub.")
    parser.add_argument("-m", "--message", default=None, help="Commit message.")
    parser.add_argument("--remote", default="origin", help="Remote name. Default: origin.")
    parser.add_argument("--branch", default=None, help="Branch name. Default: current branch.")
    parser.add_argument("--no-push", action="store_true", help="Commit only; do not push.")
    args = parser.parse_args()

    root = repo_root()
    branch = args.branch or current_branch(root)
    message = args.message or f"Update project files {datetime.now():%Y-%m-%d %H:%M}"

    print(f"Repository: {root}")
    print(f"Target: {args.remote}/{branch}")

    run(["git", "add", "-A"], root)
    if has_changes(root):
        run(["git", "commit", "-m", message], root)
    else:
        print("No changes to commit.")

    if not args.no_push:
        run(["git", "push", args.remote, branch], root)


if __name__ == "__main__":
    main()
