#!/bin/bash

# GPU and process monitoring script
# Run this in a separate terminal while your parallel processing runs

echo "=== GPU & Process Monitoring ==="
echo "Press Ctrl+C to stop monitoring"
echo ""

# Function to display GPU usage
show_gpu_usage() {
    echo "=== GPU Usage ($(date)) ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        while IFS=, read -r gpu_id name util mem_used mem_total temp; do
            echo "GPU $gpu_id ($name): ${util}% utilization, ${mem_used}MB/${mem_total}MB memory, ${temp}°C"
        done
    else
        echo "nvidia-smi not available"
    fi
    echo ""
}

# Function to display process information
show_processes() {
    echo "=== Python Processes ==="
    ps aux | grep "[p]ython.*parse-papers-parallel" | while read -r line; do
        echo "$line"
    done
    echo ""
    
    echo "=== Process Tree ==="
    pgrep -f "parse-papers-parallel" | head -5 | while read -r pid; do
        ps -o pid,ppid,cmd --pid=$pid --forest 2>/dev/null
    done
    echo ""
}

# Main monitoring loop
while true; do
    clear
    show_gpu_usage
    show_processes
    
    echo "=== System Load ==="
    uptime
    echo ""
    
    echo "Refreshing in 2 seconds... (Ctrl+C to stop)"
    sleep 2
done 