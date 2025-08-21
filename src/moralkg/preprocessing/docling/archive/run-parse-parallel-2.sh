#!/bin/bash

# Simple parallel processing script using multiprocessing with GPU acceleration
# Much simpler and more reliable than Ray

cd /opt/extra/avijit/projects/moralkg

echo "Starting parallel processing..."

# Activate conda environment
source /opt/extra/avijit/miniconda3/etc/profile.d/conda.sh
conda init
conda activate moral-kg

# Run the parallel parser
python data/scripts/phil-papers/parse-papers-parallel.py \
    --skip-existing \
    --num-workers 4 \
    --chunk-size 10 \
    --use-gpu

echo "Processing complete!" 