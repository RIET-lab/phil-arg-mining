#!/bin/bash

CODES=(
  BASDW
  BASRME-2
  BASTWO-3
  BASWWE
  GAREAM-3
  ICHECZ
  RINETF
  RINNEF
  SCHEBA-6
  SYLAEN
)

for code in "${CODES[@]}"; do
  in_file="results/${code}/${code}.json"
  echo "Cleaning $in_file"
  python ../clean_text.py "$in_file" 
done
