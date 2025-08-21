#!/bin/bash

# Script to determine what percentage of categories contain "philosophy" or "ethics"
# Usage: ./determine-precent-phil-ethics-categories.sh

# Set script directory and JSON file path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JSON_FILE="$SCRIPT_DIR/../data/metadata/2025-07-09-categories.json"

# Check if JSON file exists
if [[ ! -f "$JSON_FILE" ]]; then
    echo "Error: JSON file not found at $JSON_FILE"
    exit 1
fi

echo "Analyzing categories in: $JSON_FILE"
echo "Searching for categories containing 'philosophy' or 'ethics' (case-insensitive)..."
echo

# Extract category names from JSON (first quoted string in each array)
# The pattern looks for lines with quoted strings that are category names
# We'll extract lines that start with whitespace and a quote, which are the category names
category_names=$(grep -E '^\s*"[^"]*",' "$JSON_FILE" | \
    sed 's/^\s*"\([^"]*\)".*/\1/')

# Count total categories
total_categories=$(echo "$category_names" | wc -l)

# Count categories containing "philosophy" or "ethics" (case-insensitive)
matching_categories=$(echo "$category_names" | \
    tr '[:upper:]' '[:lower:]' | \
    grep -i -E "(philosophy|ethic|moral|value|virtue|norm|deontolog|meta)" | \
    wc -l)

# Calculate percentage using awk for floating point arithmetic
if [[ $total_categories -gt 0 ]]; then
    percentage=$(echo "$matching_categories $total_categories" | \
        awk '{printf "%.2f", ($1 * 100) / $2}')
else
    percentage="0.00"
fi

# Display results
echo "Total categories: $total_categories"
echo "Categories containing (philosophy|ethic|moral|value|virtue|norm|deontolog|meta): $matching_categories"
echo "Percentage: $percentage%"

# Show some examples of matching categories
echo
echo "Examples of matching categories:"
echo "$category_names" | \
    grep -iE "(philosophy|ethic|moral|value|virtue|norm|deontolog|meta)" | \
    head -1000

echo
echo "Analysis complete!"
