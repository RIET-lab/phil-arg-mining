#!/bin/bash

# setup.sh
# 
# Shell wrapper for the MERe Workshop setup process.

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Run the setup script
python wipe-labels.py "$@"
