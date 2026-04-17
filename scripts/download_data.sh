#!/usr/bin/env bash
set -e

FOLDER_ID="1340tEG3_bL9ojHJ8hQmMkBoZ9dSKYUhV"
DEST_DIR="$(dirname "$0")/../data"

mkdir -p "$DEST_DIR"

echo "Downloading data from Google Drive folder: $FOLDER_ID"
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "$DEST_DIR" --remaining-ok

echo "Extracting any archives..."
find "$DEST_DIR" -maxdepth 2 \( -name "*.zip" -o -name "*.tar.gz" -o -name "*.tar.bz2" -o -name "*.tar" \) | while read -r archive; do
    dir="$(dirname "$archive")"
    echo "  Extracting: $archive"
    case "$archive" in
        *.zip)      unzip -q "$archive" -d "$dir" ;;
        *.tar.gz)   tar -xzf "$archive" -C "$dir" ;;
        *.tar.bz2)  tar -xjf "$archive" -C "$dir" ;;
        *.tar)      tar -xf  "$archive" -C "$dir" ;;
    esac
done

echo "Done."