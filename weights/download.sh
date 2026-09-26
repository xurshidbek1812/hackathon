#!/usr/bin/env bash
# Fetch model weights (run once, with internet, before the offline evaluation).
# The repository already ships both files; this only restores missing ones and
# verifies the checksums.
#   yolo11m.pt  detector used by the submission (solution.py)
#   yolo11s.pt  lighter detector used by the online CPU demo only
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://github.com/ultralytics/assets/releases/download/v8.3.0"
for f in yolo11m.pt yolo11s.pt; do
  if [ ! -f "$f" ]; then
    echo "downloading $f"
    curl -L --fail -o "$f" "$BASE/$f"
  fi
done
sha256sum -c SHA256SUMS
