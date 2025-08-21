#!/bin/bash

# List of paper codes
codes=(
  BASRME-2
  BASTWO-3
  BASWWE
  GAREAM-3
  ICHECZ
  RINETF
  SCHEBA-6
  SYLAEN
)

num_gpus=4
i=0

for code in "${codes[@]}"; do
  gpu=$((i % num_gpus))
  echo "Processing $code on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python pre-trained-am-model.py -i ../../archive/initial-curation/${code}.txt -o results/${code}/${code}.json -v &
  ((i++))
  # Wait for all jobs if we've launched num_gpus jobs
  if (( i % num_gpus == 0 )); then
    wait
  fi
done

# Wait for any remaining jobs
wait

echo "All jobs finished."
