#!/bin/bash

# Simple monitor for the parallel parsing job

echo "=== Job Monitor $(date) ==="

# Job status
if pgrep -f "parse-papers-parallel.py" > /dev/null; then
    echo "Status: RUNNING"
    ps -f $(pgrep -f "parse-papers-parallel.py") | tail -n +2
else
    echo "Status: NOT RUNNING"
fi

echo ""

# GPU status
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
awk -F',' '{printf "GPU %s: %s%% util, %s/%s MB mem\n", $1, $3, $4, $5}'

echo ""

# Ray status
echo "=== Ray Status ==="
if ray status > /dev/null 2>&1; then
    ray status | grep -E "(cluster|Resources)"
else
    echo "Ray not running"
fi 