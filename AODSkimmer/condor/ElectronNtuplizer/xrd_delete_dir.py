#!/usr/bin/env python3
"""
Delete all files inside a directory stored over xrootd.
Usage: python xrootd_delete_files.py <server> <remote_dir> [--recursive] [--dry-run]
Example: python xrootd_delete_files.py root://myserver.cern.ch /store/user/mydir --recursive
"""

import subprocess
import argparse
import sys


def xrdfs(server: str, *args) -> tuple[int, str, str]:
    """Run an xrdfs command and return (returncode, stdout, stderr)."""
    cmd = ["xrdfs", server] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def list_directory(server: str, path: str) -> list[dict]:
    """
    List entries in a remote directory.
    Returns a list of dicts with keys: 'type' ('file' or 'dir'), 'path'.
    """
    rc, stdout, stderr = xrdfs(server, "ls", "-l", path)
    if rc != 0:
        print(f"[ERROR] Failed to list '{path}': {stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    entries = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # xrdfs ls -l output format:
        # d<perms> <date> <time> <size> <path>   -> directory
        # -<perms> <date> <time> <size> <path>   -> file
        parts = line.split()
        if len(parts) < 5:
            continue
        flags = parts[0]
        entry_path = parts[-1]
        entry_type = "dir" if flags.startswith("d") else "file"
        entries.append({"type": entry_type, "path": entry_path})

    return entries


def collect_files(server: str, path: str, recursive: bool) -> list[str]:
    """Collect all file paths under the given remote directory."""
    files = []
    entries = list_directory(server, path)

    for entry in entries:
        if entry["type"] == "file":
            files.append(entry["path"])
        elif entry["type"] == "dir" and recursive:
            files.extend(collect_files(server, entry["path"], recursive=True))

    return files


def delete_files(server: str, files: list[str], dry_run: bool) -> tuple[int, int]:
    """Delete a list of remote files. Returns (success_count, fail_count)."""
    success, fail = 0, 0
    for path in files:
        if dry_run:
            print(f"[DRY-RUN] Would delete: {path}")
            success += 1
            continue

        rc, _, stderr = xrdfs(server, "rm", path)
        if rc == 0:
            print(f"[DELETED] {path}")
            success += 1
        else:
            print(f"[FAILED]  {path}: {stderr.strip()}", file=sys.stderr)
            fail += 1

    return success, fail


def main():
    parser = argparse.ArgumentParser(
        description="Delete all files inside an xrootd remote directory."
    )
    #parser.add_argument("server", help="xrootd server URL, e.g. root://myserver.cern.ch")
    parser.add_argument("remote_dir", help="Remote directory path, e.g. /store/user/mydir")
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively delete files in subdirectories",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="List files that would be deleted without actually deleting them",
    )
    args = parser.parse_args()

    server = 'root://cmseos.fnal.gov/'
    remote_dir = f'/store/group/lpcmetx/iDMe/Samples/Ntuples/{args.remote_dir}'
    recursive = True
    
    print(f"Server    : {server}")
    print(f"Directory : {remote_dir}")
    print(f"Dry-run   : {args.dry_run}")
    print()

    files = collect_files(server, remote_dir, recursive=recursive)

    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} file(s) to delete.\n")
    success, fail = delete_files(server, files, dry_run=args.dry_run)

    print(f"\nDone. {success} deleted, {fail} failed.")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
