#!/usr/bin/env python3
"""
Safely remove .venv from git tracking, create a sensible .gitignore for Python ML projects,
generate requirements.txt, add & commit changes, and report total size removed.

Usage: python scripts/git_clean_venv.py [--path PATH] [--yes]

- Checks for `.venv` in the provided path (default: current directory)
- Merges a recommended `.gitignore` for Python/ML projects
- Runs `git rm -r --cached .venv` to stop tracking the venv
- Runs `pip freeze > requirements.txt`
- Adds and commits `.gitignore` and `requirements.txt` (asks for confirmation unless `--yes`)
- Prints total size of `.venv` that was removed from tracking

Works on Windows. Uses subprocess, os, shutil, pathlib.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def sizeof_fmt(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def folder_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total


def prompt_yes_no(question: str, default: bool = False) -> bool:
    yes = {"y", "yes"}
    no = {"n", "no"}
    prompt = " [y/N]: " if not default else " [Y/n]: "
    while True:
        try:
            ans = input(question + prompt).strip().lower()
        except EOFError:
            return default
        if not ans:
            return default
        if ans in yes:
            return True
        if ans in no:
            return False
        print("Please answer 'y' or 'n'.")


GITIGNORE_TEMPLATE = r"""
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
.pytest_cache/

# Jupyter Notebook checkpoints
.ipynb_checkpoints/

# pyenv
.python-version

# envs
.venv/
venv/
env/

# Virtualenv
ENV/
env.bak/

# IDEs
.vscode/
.idea/

# Data files and outputs commonly used in ML
data/
data_raw/
datasets/
models/
outputs/
checkpoints/
*.h5
*.hdf5
*.ckpt
*.pth
*.pkl
*.joblib

# Logs
logs/
*.log

# macOS
.DS_Store

# node
node_modules/

# Misc
*.bak
*.orig
"""


def merge_gitignore(root: Path) -> bool:
    gitignore_path = root / ".gitignore"
    try:
        existing_lines: List[str] = []
        if gitignore_path.exists():
            existing_lines = gitignore_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        template_lines = [l.rstrip() for l in GITIGNORE_TEMPLATE.splitlines() if l.strip() != ""]
        # Merge while preserving existing order, append any missing template entries
        set_existing = set(line.strip() for line in existing_lines if line.strip())
        merged = existing_lines[:]
        added = False
        for line in template_lines:
            if line.strip() not in set_existing:
                merged.append(line)
                added = True
        if added or not gitignore_path.exists():
            gitignore_path.write_text("\n".join(merged) + "\n", encoding="utf-8")
            print(f"Wrote/updated .gitignore at {gitignore_path}")
            return True
        else:
            print(".gitignore already contains template entries. No changes made.")
            return False
    except Exception as e:
        print("Failed to write .gitignore:", e)
        return False


def check_git_available() -> bool:
    try:
        res = subprocess.run(["git", "--version"], capture_output=True, text=True)
        return res.returncode == 0
    except Exception:
        return False


def run_command(cmd: List[str], cwd: Path = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except Exception as e:
        class Dummy:
            returncode = 1
            stdout = ""
            stderr = str(e)
        return Dummy()


def main():
    parser = argparse.ArgumentParser(description="Remove .venv from git tracking, add .gitignore and requirements.txt")
    parser.add_argument("--path", "-p", default=".", help="Repository root path (default: current directory)")
    parser.add_argument("--yes", "-y", action="store_true", help="Assume yes to all prompts")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    print("\n=== Git .venv clean helper ===\n")
    print("Repository root:", root)

    if not root.exists():
        print("Path does not exist:", root)
        sys.exit(1)

    venv_path = root / ".venv"
    if not venv_path.exists() or not venv_path.is_dir():
        print("No .venv folder found at:", venv_path)
        # Offer to check common alternatives
        alt = [root / "venv", root / "env"]
        found_alt = [p for p in alt if p.exists() and p.is_dir()]
        if found_alt:
            print("Found alternative virtual envs:")
            for p in found_alt:
                print(" -", p)
        else:
            print("Nothing to remove. Exiting.")
        return

    # Compute size before any git changes
    size = folder_size(venv_path)
    print(f"Size of .venv: {sizeof_fmt(size)}")

    # Ensure git exists
    if not check_git_available():
        print("git not available in PATH. Please install Git and ensure it's on PATH.")
        sys.exit(1)

    # Merge/create .gitignore
    try:
        merge_gitignore(root)
    except Exception as e:
        print("Error while updating .gitignore:", e)

    # Remove .venv from git tracking
    print("\nRemoving .venv from git tracking (git rm --cached)")
    if not args.yes:
        if not prompt_yes_no("Proceed with git rm --cached .venv ?"):
            print("Aborted by user.")
            return

    res = run_command(["git", "rm", "-r", "--cached", ".venv"], cwd=root)
    if res.returncode != 0:
        print("git rm returned non-zero:", res.stderr.strip())
    else:
        print("git rm --cached .venv output:\n", res.stdout.strip())

    # Generate requirements.txt
    print("\nGenerating requirements.txt using pip freeze")
    try:
        proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        if proc.returncode != 0:
            print("pip freeze failed:", proc.stderr)
        else:
            req_file = root / "requirements.txt"
            req_file.write_text(proc.stdout)
            print(f"Wrote {req_file}")
    except Exception as e:
        print("Failed to run pip freeze:", e)

    # Add and commit changes
    print("\nStaging .gitignore and requirements.txt")
    add_res = run_command(["git", "add", ".gitignore", "requirements.txt"], cwd=root)
    if add_res.returncode != 0:
        print("git add failed:", add_res.stderr.strip())

    # Check staged changes
    diff = run_command(["git", "diff", "--staged", "--name-only"], cwd=root)
    staged_files = diff.stdout.strip().splitlines() if diff.returncode == 0 else []
    if not staged_files:
        print("No staged changes to commit.")
    else:
        print("Staged files:")
        for f in staged_files:
            print(" -", f)
        if not args.yes:
            if not prompt_yes_no("Commit these changes?"):
                print("Skipping commit.")
                # still report size and exit
                print(f"Total size of .venv removed from tracking: {sizeof_fmt(size)}")
                return
        commit_msg = "Remove .venv from repo, add .gitignore and requirements.txt"
        commit_res = run_command(["git", "commit", "-m", commit_msg], cwd=root)
        if commit_res.returncode != 0:
            print("git commit failed:", commit_res.stderr.strip())
        else:
            print("Committed changes:\n", commit_res.stdout.strip())

    print(f"\nTotal size of .venv removed from git tracking: {sizeof_fmt(size)}")
    print("Done.")


if __name__ == "__main__":
    main()
