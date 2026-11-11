#!/usr/bin/env bash
# Regenerate the submission from this repository and compare it, byte for byte, with a CSV you
# already have. Prints only SAME or DIFFERENT, nothing about the match.
#   ./verify_reproduction.sh /path/to/Garuda.csv
set -euo pipefail
cd "$(dirname "$0")"
[ -n "${MTAH_DATA_DIR:-}" ] || { echo "Set MTAH_DATA_DIR to the folder that contains tactical/"; exit 1; }
TMP=$(mktemp -d)
.venv/bin/python garuda/scripts/predict.py --out "$TMP/Garuda.csv" > /dev/null
if cmp -s "$TMP/Garuda.csv" "$1"; then echo "SAME"; else echo "DIFFERENT"; fi
rm -rf "$TMP"
