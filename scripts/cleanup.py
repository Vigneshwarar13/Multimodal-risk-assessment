#!/usr/bin/env python3
"""
Project cleanup utility (Windows-compatible)

Features:
- Scan project tree recursively and list files larger than threshold (default 100MB)
- Find common unnecessary folders (__pycache__, .ipynb_checkpoints, venv, env, .venv, node_modules)
- Compute total size of each such folder
- Ask user confirmation before deleting each folder
- Delete selected folders safely (handles read-only files)
- Optionally generate requirements.txt using installed packages (uses current Python executable)
- Print total space freed after cleanup

Usage: python scripts/cleanup.py [--path PATH] [--threshold MB]
"""

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

# Constants
DEFAULT_THRESHOLD_MB = 100
UNNECESSARY_NAMES = {"__pycache__", ".ipynb_checkpoints", "venv", "env", ".venv", "node_modules"}

# Helpers

def sizeof_fmt(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def folder_size(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except Exception:
        pass
    return total


def find_large_files(root: Path, threshold_bytes: int) -> List[Tuple[Path, int]]:
    found = []
    for p in root.rglob("*"):
        if p.is_file():
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size >= threshold_bytes:
                found.append((p, size))
    return sorted(found, key=lambda x: x[1], reverse=True)


def find_unnecessary_folders(root: Path) -> List[Path]:
    matches = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name in UNNECESSARY_NAMES:
            matches.append(p)
    # Deduplicate and sort by path
    unique_paths = sorted({p.resolve() for p in matches})
    return [Path(p) for p in unique_paths]


def handle_remove_readonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue is not None:
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass
    else:
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass


def safe_rmtree(path: Path) -> bool:
    try:
        shutil.rmtree(path, onerror=handle_remove_readonly)
        return True
    except Exception as e:
        print(f"Failed to remove {path} — {e}")
        return False


def prompt_yes_no(question: str, default: bool = False) -> bool:
    yes = {"y", "yes"}
    no = {"n", "no"}
    prompt = " [y/N]: " if not default else " [Y/n]: "
    while True:
        ans = input(question + prompt).strip().lower()
        if not ans:
            return default
        if ans in yes:
            return True
        if ans in no:
            return False
        print("Please answer 'y' or 'n'.")


def generate_requirements(output_path: Path) -> bool:
    try:
        print("Generating requirements.txt using pip freeze...")
        res = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        if res.returncode != 0:
            print("pip freeze failed:", res.stderr)
            return False
        output_path.write_text(res.stdout)
        print(f"Wrote {output_path}")
        return True
    except Exception as e:
        print("Failed to generate requirements.txt:", e)
        return False


# CLI

def main():
    parser = argparse.ArgumentParser(description="Project cleanup utility")
    parser.add_argument("--path", "-p", default=".", help="Root path to scan (default: current directory)")
    parser.add_argument("--threshold", "-t", type=int, default=DEFAULT_THRESHOLD_MB, help=f"Large file threshold in MB (default: {DEFAULT_THRESHOLD_MB})")
    parser.add_argument("--yes", "-y", action="store_true", help="Assume yes to all deletions (use carefully)")
    parser.add_argument("--dry-run", action="store_true", help="Don't perform deletions; only show what would be deleted")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        print("Path does not exist:", root)
        return

    print("\n=== Project Cleanup Utility ===\n")
    print("Scanning:", root)

    threshold_bytes = args.threshold * 1024 * 1024

    print("\n-- Large files (>{} MB) --".format(args.threshold))
    large_files = find_large_files(root, threshold_bytes)
    if not large_files:
        print("No large files found.")
    else:
        for p, size in large_files:
            print(f"{sizeof_fmt(size):>9}  {p}")

    print("\n-- Common unnecessary folders --")
    folders = find_unnecessary_folders(root)
    if not folders:
        print("No common unnecessary folders found.")
    else:
        folder_infos = []
        for f in folders:
            size = folder_size(f)
            folder_infos.append((f, size))
        folder_infos.sort(key=lambda x: x[1], reverse=True)
        for idx, (f, size) in enumerate(folder_infos, start=1):
            print(f"{idx:2d}. {sizeof_fmt(size):>9}  {f}")

        # Ask about deletion
        total_freed = 0
        potential_freed = 0
        for f, size in folder_infos:
            if args.yes:
                do_delete = True
            else:
                do_delete = prompt_yes_no(f"Delete folder {f} (size {sizeof_fmt(size)})?")
            if do_delete:
                if args.dry_run:
                    print(f"Dry-run: would remove {f} ({sizeof_fmt(size)})")
                    potential_freed += size
                else:
                    print(f"Removing {f}...")
                    success = safe_rmtree(f)
                    if success:
                        total_freed += size
                        print(f"Removed {f} ({sizeof_fmt(size)})")
                    else:
                        print(f"Failed to remove {f}")
            else:
                print(f"Skipped {f}")

        if args.dry_run:
            print("\nPotential space that would be freed (dry-run):", sizeof_fmt(potential_freed))
        else:
            print("\nTotal space freed from deleted folders:", sizeof_fmt(total_freed))

    # Optionally generate requirements.txt
    req_path = root / "requirements.txt"
    if prompt_yes_no("Generate requirements.txt from installed packages?"):
        if args.dry_run:
            print(f"Dry-run: would generate {req_path}")
        else:
            generate_requirements(req_path)

    print("\nScan complete.")


if __name__ == "__main__":
    main()
