#!/bin/bash

# Usage: ./clean_texts.sh [options] input_path [output_path]

set -e # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILE_NAME="clean_texts.py"
CLEANER_SCRIPT="$SCRIPT_DIR/$FILE_NAME"

# Check if clean_texts.py exists
if [ ! -f "$CLEANER_SCRIPT" ]; then
    echo "[ERROR] $FILE_NAME not found in $SCRIPT_DIR"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is required but not found"
    exit 1
fi

# Pass all arguments directly to the Python script
exec python3 "$CLEANER_SCRIPT" "$@"
