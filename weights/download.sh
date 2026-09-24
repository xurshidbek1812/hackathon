#!/usr/bin/env bash
# Fetch model weights (run once, with internet, before the offline evaluation).
# The repository already ships weights/yolo11m.pt; this script only restores it
# if it is missing and verifies the checksum.
set -euo pipefail
cd "$(dirname "$0")"
URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
if [ ! -f yolo11m.pt ]; then
  echo "downloading yolo11m.pt"
  curl -L --fail -o yolo11m.pt "$URL"
fi
sha256sum -c SHA256SUMS
