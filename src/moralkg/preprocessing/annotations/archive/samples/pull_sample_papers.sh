#!/bin/bash

MORALKG_DIR=/opt/extra/avijit/projects/moralkg

# Create sample-papers directory if it doesn't exist
mkdir -p $MORALKG_DIR/data/sample-papers

# Read codes file and copy matching files
while IFS= read -r code || [ -n "$code" ]; do
    # Skip empty lines
    [ -z "$code" ] && continue
    
    # Check if any matching files exist
    if ls $MORALKG_DIR/data/docling/"$code".* 1>/dev/null 2>&1; then
        echo "Copying files for $code..."
        cp $MORALKG_DIR/data/docling/"$code".* $MORALKG_DIR/data/sample-papers/ 2>/dev/null
    else
        echo "Warning: No files found for $code"
    fi
done < $MORALKG_DIR/data/sample-papers/codes

echo "Sample papers have been copied to the sampling directory" 