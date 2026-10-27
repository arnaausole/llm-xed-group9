#!/usr/bin/env bash
# Download the XED dataset (CC-BY 4.0) into data/raw/XED.
# The data itself is NOT committed to this repository.
set -euo pipefail

DEST="data/raw/XED"
# Pin the exact upstream commit for reproducibility.
XED_COMMIT="76666d2"

if [ -d "$DEST/.git" ]; then
    echo "XED already present in $DEST"
else
    mkdir -p data/raw
    git clone https://github.com/Helsinki-NLP/XED.git "$DEST"
fi

git -C "$DEST" checkout -q "$XED_COMMIT"
echo "XED checked out at commit $(git -C "$DEST" rev-parse --short HEAD)"
