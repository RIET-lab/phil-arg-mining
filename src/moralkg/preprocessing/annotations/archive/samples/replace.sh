#!/bin/bash

# Helper script to interactively replace papers in a sample
# Usage: ./replace.sh -n SAMPLE_SIZE

set -e

# Default values
SAMPLE_SIZE=""
SEED=42
SAMPLE_DIR=""

# Function to show usage
usage() {
    echo "Usage: $0 -n SAMPLE_SIZE [options]"
    echo ""
    echo "Options:"
    echo "  -n, --sample-size SIZE    Sample size (required)"
    echo "  -s, --seed SEED          Random seed (default: 42)"
    echo "  -d, --sample-dir DIR     Sample directory (default: data/annotations/samples/nSIZE)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "This script helps you interactively replace papers in your sample."
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -d|--sample-dir)
            SAMPLE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SAMPLE_SIZE" ]]; then
    echo "Error: Sample size (-n) is required"
    usage
fi

# Set default sample directory if not provided
if [[ -z "$SAMPLE_DIR" ]]; then
    SAMPLE_DIR="data/annotations/samples/n${SAMPLE_SIZE}"
fi

# Check if sample directory exists
if [[ ! -d "$SAMPLE_DIR" ]]; then
    echo "Error: Sample directory does not exist: $SAMPLE_DIR"
    echo "Run the sampling pipeline first with: ./sample.sh -n $SAMPLE_SIZE"
    exit 1
fi

# Check if sample.csv exists
if [[ ! -f "$SAMPLE_DIR/sample.csv" ]]; then
    echo "Error: Sample file not found: $SAMPLE_DIR/sample.csv"
    exit 1
fi

echo "=== INTERACTIVE PAPER REPLACEMENT ==="
echo "Sample directory: $SAMPLE_DIR"
echo "Sample size: $SAMPLE_SIZE"
echo "Seed: $SEED"
echo ""

# Show current sample
echo "Current papers in sample:"
python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$SAMPLE_DIR/sample.csv')
    for i, paper_id in enumerate(df['identifier'], 1):
        print(f'  {i:2d}. {paper_id}')
except Exception as e:
    print(f'Error reading sample: {e}')
    sys.exit(1)
" || exit 1

echo ""

# Interactive loop for replacements
while true; do
    echo "Enter a paper identifier to replace (or 'quit' to exit):"
    read -r PAPER_ID
    
    if [[ "$PAPER_ID" == "quit" ]] || [[ "$PAPER_ID" == "q" ]] || [[ "$PAPER_ID" == "exit" ]]; then
        echo "Exiting replacement tool."
        break
    fi
    
    if [[ -z "$PAPER_ID" ]]; then
        echo "Please enter a paper identifier."
        continue
    fi
    
    # Check if paper exists in sample
    PAPER_EXISTS=$(python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$SAMPLE_DIR/sample.csv')
    if '$PAPER_ID' in df['identifier'].values:
        print('yes')
    else:
        print('no')
except Exception:
    print('error')
")
    
    if [[ "$PAPER_EXISTS" == "no" ]]; then
        echo "Error: Paper '$PAPER_ID' not found in current sample."
        echo "Available papers:"
        python -c "
import pandas as pd
try:
    df = pd.read_csv('$SAMPLE_DIR/sample.csv')
    for paper_id in df['identifier']:
        print(f'  {paper_id}')
except Exception:
    pass
"
        continue
    elif [[ "$PAPER_EXISTS" == "error" ]]; then
        echo "Error reading sample file."
        continue
    fi
    
    # Run sampling-replace.py
    echo ""
    echo "Running replacement tool for paper: $PAPER_ID"
    echo "----------------------------------------"
    
    python /opt/extra/avijit/projects/moralkg/data/scripts/annotations/samples/sampling-replace.py \
        -n "$SAMPLE_SIZE" \
        -p "$PAPER_ID" \
        -s "$SEED" \
        -d "$SAMPLE_DIR"
    
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "Replacement completed successfully!"
        echo ""
        echo "Updated sample:"
        python -c "
import pandas as pd
try:
    df = pd.read_csv('$SAMPLE_DIR/sample.csv')
    for i, paper_id in enumerate(df['identifier'], 1):
        print(f'  {i:2d}. {paper_id}')
except Exception as e:
    print(f'Error reading updated sample: {e}')
"
    else
        echo "Replacement failed or was cancelled."
    fi
    
    echo ""
done

echo ""
echo "Final sample available in: $SAMPLE_DIR/sample.csv" 
