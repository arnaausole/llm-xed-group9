#!/usr/bin/env bash
# Download XED into data/raw/XED (not in git)
set -euo pipefail

DEST="data/raw/XED"
# fixed commit so everyone gets the same data
XED_COMMIT="76666d2"

if [ -d "$DEST/.git" ]; then
    echo "XED already present in $DEST"
else
    mkdir -p data/raw
    git clone https://github.com/Helsinki-NLP/XED.git "$DEST"
fi

git -C "$DEST" checkout -q "$XED_COMMIT"
echo "XED checked out at commit $(git -C "$DEST" rev-parse --short HEAD)"
