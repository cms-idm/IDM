#!/bin/bash
# Recursively merge all files from an EOS source directory tree into an EOS
# destination directory tree that may already contain files/subdirectories,
# without clobbering existing destination files unless explicitly forced.
#
# The trees are laid out as:
#   <SRC>/2024/<process>/<subprocess>/*.root
#   <DST>/2024/<process>/<subprocess>/*.root
# Not every <process>/<subprocess> combination necessarily exists on both
# sides — missing destination subdirectories are created as needed.
#
# Usage:
#   ./rename_eos_bkg.sh [--dry-run] [--force] [--rmdir]
#
#   --dry-run   print the eos mkdir/mv commands that would run, but do nothing
#   --force     overwrite destination files whose relative path collides
#   --rmdir     after a successful move, remove now-empty directories (and
#               SRC itself) left behind under the source tree
#
# Without --force, any file whose path relative to SRC/DST collides is
# skipped and reported at the end so nothing is silently overwritten.

set -euo pipefail

export EOS_MGM_URL="${EOS_MGM_URL:-root://cmseos.fnal.gov}"

SRC="/store/group/lpcmetx/iDMe/Samples/Ntuples/background_Jul2026noID_slim"
DST="/store/group/lpcmetx/iDMe/Samples/Ntuples/background_Aug2026noIDSep_slim"

DRY_RUN=0
FORCE=0
DO_RMDIR=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --force)   FORCE=1 ;;
        --rmdir)   DO_RMDIR=1 ;;
        *)
            echo "Unknown option: $arg" >&2
            echo "Usage: $0 [--dry-run] [--force] [--rmdir]" >&2
            exit 1
            ;;
    esac
done

if ! eos ls "$SRC" >/dev/null 2>&1; then
    echo "Source directory not found or not accessible: $SRC" >&2
    exit 1
fi

if ! eos ls "$DST" >/dev/null 2>&1; then
    echo "Destination directory does not exist, creating it: $DST"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] eos mkdir -p $DST"
    else
        eos mkdir -p "$DST"
    fi
fi

SRC_BASE="$(basename "$SRC")"
DST_BASE="$(basename "$DST")"

echo "Scanning source tree: $SRC"
mapfile -t SRC_FILES < <(eos find -f "$SRC")

echo "Scanning destination tree: $DST"
mapfile -t DST_FILES < <(eos find -f "$DST" 2>/dev/null)

if [[ "${#SRC_FILES[@]}" -eq 0 ]]; then
    echo "No files found under $SRC."
    if [[ "$DRY_RUN" -eq 0 && "$DO_RMDIR" -eq 1 ]]; then
        echo "Removing empty directories under: $SRC"
        mapfile -t SRC_DIRS < <(eos find -d "$SRC" 2>/dev/null | sort -r)
        for d in "${SRC_DIRS[@]}"; do
            eos rmdir "$d" 2>/dev/null || true
        done
        eos rmdir "$SRC" 2>/dev/null || true
    else
        echo "Nothing to do. Rerun with --rmdir to remove the empty directory tree, or clean it up manually."
    fi
    exit 0
fi

declare -A DST_SET
for f in "${DST_FILES[@]}"; do
    rel="${f#*"$DST_BASE"/}"
    DST_SET["$rel"]=1
done

declare -A NEEDED_DIRS
declare -A TO_MOVE

for f in "${SRC_FILES[@]}"; do
    rel="${f#*"$SRC_BASE"/}"
    if [[ -n "${DST_SET[$rel]:-}" && "$FORCE" -eq 0 ]]; then
        continue
    fi
    TO_MOVE["$rel"]=1
    NEEDED_DIRS["$(dirname "$rel")"]=1
done

skipped=()
for f in "${SRC_FILES[@]}"; do
    rel="${f#*"$SRC_BASE"/}"
    if [[ -z "${TO_MOVE[$rel]:-}" ]]; then
        skipped+=("$rel")
    fi
done

echo ""
echo "Found ${#SRC_FILES[@]} file(s) under source; ${#TO_MOVE[@]} to move, ${#skipped[@]} skipped (already present at destination)."

for d in "${!NEEDED_DIRS[@]}"; do
    destdir="$DST/$d"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] eos mkdir -p $destdir"
    else
        eos mkdir -p "$destdir" >/dev/null 2>&1 || true
    fi
done

moved=0
for rel in "${!TO_MOVE[@]}"; do
    srcfile="$SRC/$rel"
    destfile="$DST/$rel"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry-run] eos mv \"$srcfile\" \"$destfile\""
    else
        echo "Moving $rel"
        eos mv "$srcfile" "$destfile"
    fi
    moved=$((moved + 1))
done

echo ""
echo "Moved: $moved file(s)"

if [[ "${#skipped[@]}" -gt 0 ]]; then
    echo "Skipped ${#skipped[@]} file(s) that already exist at the destination path (rerun with --force to overwrite):"
    printf '  %s\n' "${skipped[@]}"
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "Dry run complete — no files were actually moved."
    exit 0
fi

remaining=$(eos find -f "$SRC" 2>/dev/null | wc -l)
echo ""
echo "Files remaining under $SRC: $remaining"

if [[ "$remaining" -eq 0 && "$DO_RMDIR" -eq 1 ]]; then
    echo "Removing empty directories under: $SRC"
    mapfile -t SRC_DIRS < <(eos find -d "$SRC" 2>/dev/null | sort -r)
    for d in "${SRC_DIRS[@]}"; do
        eos rmdir "$d" 2>/dev/null || true
    done
    eos rmdir "$SRC" 2>/dev/null || true
elif [[ "$remaining" -eq 0 ]]; then
    echo "Source tree has no files left. Rerun with --rmdir to remove the empty directory tree, or clean it up manually."
fi
