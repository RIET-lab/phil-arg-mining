#!/bin/bash

# Simple wrapper to run parallel PDF parsing with Ray

set -e

PROJECT_DIR="/opt/extra/avijit/projects/moralkg"
CONDA_ENV="moral-kg"

cd "$PROJECT_DIR"

# Check if already running
if pgrep -f "parse-papers-parallel.py" > /dev/null; then
    echo "ERROR: Job already running"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# Check packages
python -c "import ray, docling" || {
    echo "ERROR: Missing packages. Install with: pip install ray[default] docling"
    exit 1
}

# Show GPU status
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)"

# Run the job
echo "Starting parallel processing..."
python data/scripts/phil-papers/parse-papers-parallel.py "$@" 