#!/bin/bash

# Simple parallel processing script using multiprocessing with GPU acceleration
# Same as v2, but using venv instead of conda

cd /opt/extra/avijit/projects/moralkg

echo "Starting parallel processing..."

# Activate virtual environment
source /opt/extra/avijit/projects/moralkg/.venv/bin/activate

# Run the parallel parser
python data/scripts/phil-papers/parse-papers-parallel.py \
    --skip-existing \
    --num-workers 4 \
    --chunk-size 10 \
    --use-gpu

echo "Processing complete!" 