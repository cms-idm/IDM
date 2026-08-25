#!/usr/bin/env python3
"""
Rename/move a directory of ntuples stored over xrootd (e.g. an EOS namespace rename).
Usage: python xrd_rename_dir.py <old_name> <new_name> [--dry-run]
Example: python xrd_rename_dir.py background_Jul2026noID background_Aug2026noID
"""

import subprocess
import argparse
import sys


def xrdfs(server: str, *args) -> tuple[int, str, str]:
    """Run an xrdfs command and return (returncode, stdout, stderr)."""
    cmd = ["xrdfs", server] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def exists(server: str, path: str) -> bool:
    rc, _, _ = xrdfs(server, "stat", path)
    return rc == 0


def main():
    parser = argparse.ArgumentParser(
        description="Rename a directory in /store/group/lpcmetx/iDMe/Samples/Ntuples/ over xrootd."
    )
    parser.add_argument("old_name", help="Current directory name, e.g. background_Jul2026noID")
    parser.add_argument("new_name", help="New directory name, e.g. background_Aug2026noID")
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print what would be done without actually renaming",
    )
    args = parser.parse_args()

    server = 'root://cmseos.fnal.gov/'
    base = '/store/group/lpcmetx/iDMe/Samples/Ntuples'
    old_path = f'{base}/{args.old_name}'
    new_path = f'{base}/{args.new_name}'

    print(f"Server : {server}")
    print(f"From   : {old_path}")
    print(f"To     : {new_path}")
    print(f"Dry-run: {args.dry_run}")
    print()

    if not exists(server, old_path):
        print(f"[ERROR] Source directory does not exist: {old_path}", file=sys.stderr)
        sys.exit(1)

    if exists(server, new_path):
        print(f"[ERROR] Destination already exists: {new_path}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"[DRY-RUN] Would rename {old_path} -> {new_path}")
        return

    rc, _, stderr = xrdfs(server, "mv", old_path, new_path)
    if rc != 0:
        print(f"[FAILED] {stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    print(f"[RENAMED] {old_path} -> {new_path}")


if __name__ == "__main__":
    main()
