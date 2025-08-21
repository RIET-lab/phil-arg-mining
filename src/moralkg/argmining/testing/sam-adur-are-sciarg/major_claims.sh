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
  echo "Processing $code"
  python major_claims_extraction.py "$code"
done
