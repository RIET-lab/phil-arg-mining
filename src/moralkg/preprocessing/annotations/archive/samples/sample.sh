#!/bin/bash

# Script to run sampling.py followed by sampling-suggestions.py
# with the same sample-size and seed arguments

set -e  # Exit on any error

# Default values
SAMPLE_SIZE=""
SEED=42
INPUT_FILE="data/metadata/2025-07-09-en-combined-metadata.csv"
OUTPUT_DIR=""
ALLOW_AUTHOR_REPEATS=false
ALLOW_YEAR_REPEATS=false
ALLOW_CATEGORY_REPEATS=false

# Function to show usage
usage() {
    echo "Usage: $0 -n SAMPLE_SIZE [options]"
    echo ""
    echo "Options:"
    echo "  -n, --sample-size SIZE       Number of papers to sample (required)"
    echo "  -s, --seed SEED              Random seed for reproducibility (default: 42)"
    echo "  -i, --input-file FILE        Path to input CSV file (default: data/metadata/2025-07-09-en-combined-metadata.csv)"
    echo "  -o, --output-dir DIR         Output directory path (default: data/annotations/samples/nSIZE)"
    echo "  -a, --allow-author-repeats   Allow authors to repeat in sample (default: no repeats)"
    echo "  -y, --allow-year-repeats     Allow years to repeat in sample (default: no repeats)"
    echo "  -c, --allow-category-repeats Allow categories to repeat in sample (default: no repeats)"
    echo "  -h, --help                   Show this help message"
    echo ""
    echo "This script runs sampling.py first, then sampling-suggestions.py to supplement the sample."
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
        -i|--input-file)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -a|--allow-author-repeats)
            ALLOW_AUTHOR_REPEATS=true
            shift
            ;;
        -y|--allow-year-repeats)
            ALLOW_YEAR_REPEATS=true
            shift
            ;;
        -c|--allow-category-repeats)
            ALLOW_CATEGORY_REPEATS=true
            shift
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

# Set default output directory if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="data/annotations/samples/n${SAMPLE_SIZE}"
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=== RUNNING SAMPLE PIPELINE ==="
echo "Sample size: $SAMPLE_SIZE"
echo "Seed: $SEED"
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Allow author repeats: $ALLOW_AUTHOR_REPEATS"
echo "Allow year repeats: $ALLOW_YEAR_REPEATS"
echo "Allow category repeats: $ALLOW_CATEGORY_REPEATS"
echo ""

# Build arguments for sampling.py
SAMPLING_ARGS="-n $SAMPLE_SIZE -s $SEED -i '$INPUT_FILE' -o '$OUTPUT_DIR'"

if [[ "$ALLOW_AUTHOR_REPEATS" == "true" ]]; then
    SAMPLING_ARGS="$SAMPLING_ARGS -a"
fi

if [[ "$ALLOW_YEAR_REPEATS" == "true" ]]; then
    SAMPLING_ARGS="$SAMPLING_ARGS -y"
fi

if [[ "$ALLOW_CATEGORY_REPEATS" == "true" ]]; then
    SAMPLING_ARGS="$SAMPLING_ARGS -c"
fi

# Step 1: Run initial sampling
echo "Step 1: Running initial sampling..."
echo "Command: python sampling.py $SAMPLING_ARGS"
eval "python /opt/extra/avijit/projects/moralkg/data/scripts/annotations/samples/sampling.py $SAMPLING_ARGS"

if [[ $? -ne 0 ]]; then
    echo "Error: Initial sampling failed"
    exit 1
fi

echo ""
echo "Step 1 completed successfully"
echo ""

# Step 2: Run supplementation with suggestions
echo "Step 2: Supplementing sample with suggestions..."
echo "Command: python sampling-suggestions.py -n $SAMPLE_SIZE -s $SEED -d '$OUTPUT_DIR'"
python /opt/extra/avijit/projects/moralkg/data/scripts/annotations/samples/sampling-suggestions.py -n "$SAMPLE_SIZE" -s "$SEED" -d "$OUTPUT_DIR"

if [[ $? -ne 0 ]]; then
    echo "Error: Sample supplementation failed"
    exit 1
fi

echo ""
echo "Step 2 completed successfully"
echo ""

# Step 3: Optional paper replacement
echo "Step 3: Paper replacement (optional)..."
echo "To replace specific papers in your sample, run:"
echo "  python sampling-replace.py -n $SAMPLE_SIZE -p PAPER_ID -s $SEED -d '$OUTPUT_DIR'"
echo ""
echo "Available papers in sample:"
python -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$OUTPUT_DIR/sample.csv')
    for i, paper_id in enumerate(df['identifier'], 1):
        print(f'  {i:2d}. {paper_id}')
except Exception as e:
    print(f'Error reading sample: {e}')
    sys.exit(1)
"

echo ""
echo "=== PIPELINE COMPLETED ==="
echo "Final sample available in: $OUTPUT_DIR/sample.csv"
echo "Logs available in: $OUTPUT_DIR/sampling.log"
echo ""
echo "Next steps:"
echo "1. Review the sample in: $OUTPUT_DIR/sample.csv"
echo "2. If needed, replace papers using sampling-replace.py"
echo "3. If needed, add more papers using sampling-add.py"
echo "4. Start annotation process"
echo ""
echo "Tools available:"
echo "• Replace papers: python sampling-replace.py -n $SAMPLE_SIZE -p PAPER_ID"
echo "• Add papers: python sampling-add.py -n $SAMPLE_SIZE --add COUNT [--philosophy-only]"
echo "• Interactive replacement: ./replace.sh -n $SAMPLE_SIZE"
