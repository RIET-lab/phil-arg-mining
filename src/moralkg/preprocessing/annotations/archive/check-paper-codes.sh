#!/usr/bin/env bash

# This script checks to see if the files listed in moralkg/etc/known-paper-codes
# (i.e. those Aidan can audit well) have been downloaded to moralkg/data/pdfs and
# processed by docling into moralkg/data/docling.

DOCLING_DIR="/opt/extra/avijit/projects/moralkg/data/docling"
PDFS_DIR="/opt/extra/avijit/projects/moralkg/data/pdfs"

files=(
  BASDW
  BASRME-2
  BASTWO-3
  BASWWE
  ICHECZ
  SCHEBA-6
  GAREAM-3
  RINETF
  RINNEF
  SYLAEN
)

# Loop through and test each file
for fname in "${files[@]}"; do
  echo "Checking ${fname}:"
  
  # Check docling directory
  if [ -e "${DOCLING_DIR}/${fname}" ] || compgen -G "${DOCLING_DIR}/${fname}.*" > /dev/null; then
    echo "  DOCLING : FOUND"
  else
    echo "  DOCLING : MISSING"
  fi
  
  # Check pdfs directory
  if [ -e "${PDFS_DIR}/${fname}" ] || compgen -G "${PDFS_DIR}/${fname}.*" > /dev/null; then
    echo "  PDFS    : FOUND"
  else
    echo "  PDFS    : MISSING"
  fi
  
  echo ""
done
