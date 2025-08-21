#!/bin/bash

# Script to determine what percentage of papers have more than 100, 500, and 1000 words
# Usage: ./paper-length-splits.sh

# Set script directory and docling directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCLING_DIR="$SCRIPT_DIR/../data/docling"

# Check if docling directory exists
if [[ ! -d "$DOCLING_DIR" ]]; then
    echo "Error: Docling directory not found at $DOCLING_DIR"
    exit 1
fi

echo "Analyzing paper lengths in: $DOCLING_DIR"
echo "Searching for .md files first, then .txt files (excluding .doctags.txt)..."
echo

# Create temporary files for processing
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

PAPER_COUNTS="$TEMP_DIR/paper_counts.txt"
WORD_COUNTS="$TEMP_DIR/word_counts.txt"

# Find all unique identifiers by looking at file stems
# Get all .md and .txt files, extract stems, sort and unique
find "$DOCLING_DIR" -type f \( -name "*.md" -o -name "*.txt" \) ! -name "*.doctags.txt" | \
    sed 's/.*\///g' | \
    sed 's/\.[^.]*$//' | \
    sort -u > "$TEMP_DIR/identifiers.txt"

total_papers=0
papers_over_100=0
papers_over_500=0
papers_over_1000=0

echo "Processing papers..."

# Process each identifier
while IFS= read -r identifier; do
    if [[ -z "$identifier" ]]; then
        continue
    fi
    
    # Look for .md file first, then .txt file
    content_file=""
    if [[ -f "$DOCLING_DIR/${identifier}.md" ]]; then
        content_file="$DOCLING_DIR/${identifier}.md"
    elif [[ -f "$DOCLING_DIR/${identifier}.txt" ]]; then
        content_file="$DOCLING_DIR/${identifier}.txt"
    fi
    
    if [[ -n "$content_file" ]]; then
        # Count words in the file
        word_count=$(wc -w < "$content_file" 2>/dev/null || echo "0")
        
        # Increment counters
        ((total_papers++))
        echo "$identifier:$word_count" >> "$PAPER_COUNTS"
        echo "$word_count" >> "$WORD_COUNTS"
        
        if [[ $word_count -gt 100 ]]; then
            ((papers_over_100++))
        fi
        if [[ $word_count -gt 500 ]]; then
            ((papers_over_500++))
        fi
        if [[ $word_count -gt 1000 ]]; then
            ((papers_over_1000++))
        fi
        
        # Progress indicator
        if [[ $((total_papers % 1000)) -eq 0 ]]; then
            echo "  Processed $total_papers papers..."
        fi
    fi
done < "$TEMP_DIR/identifiers.txt"

echo "  Processed $total_papers papers total."
echo

# Calculate percentages using awk for floating point arithmetic
if [[ $total_papers -gt 0 ]]; then
    percentage_100=$(echo "$papers_over_100 $total_papers" | \
        awk '{printf "%.2f", ($1 * 100) / $2}')
    percentage_500=$(echo "$papers_over_500 $total_papers" | \
        awk '{printf "%.2f", ($1 * 100) / $2}')
    percentage_1000=$(echo "$papers_over_1000 $total_papers" | \
        awk '{printf "%.2f", ($1 * 100) / $2}')
else
    percentage_100="0.00"
    percentage_500="0.00"
    percentage_1000="0.00"
fi

# Calculate basic statistics
if [[ -s "$WORD_COUNTS" ]]; then
    mean_words=$(awk '{sum+=$1; count++} END {printf "%.0f", sum/count}' "$WORD_COUNTS")
    median_words=$(sort -n "$WORD_COUNTS" | awk '{a[NR]=$0} END {print (NR%2==1) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}')
    min_words=$(sort -n "$WORD_COUNTS" | head -1)
    max_words=$(sort -n "$WORD_COUNTS" | tail -1)
else
    mean_words="0"
    median_words="0"
    min_words="0"
    max_words="0"
fi

# Display results
echo "Paper Length Analysis Results"
echo "============================="
echo "Total papers analyzed: $total_papers"
echo
echo "Word Count Thresholds:"
echo "Papers > 100 words:  $papers_over_100 ($percentage_100%)"
echo "Papers > 500 words:  $papers_over_500 ($percentage_500%)"
echo "Papers > 1000 words: $papers_over_1000 ($percentage_1000%)"
echo
echo "Basic Statistics:"
echo "Mean word count:   $mean_words words"
echo "Median word count: $median_words words"
echo "Min word count:    $min_words words"
echo "Max word count:    $max_words words"

# Show examples of papers in different categories
echo
echo "Examples of papers by word count category:"
echo "=========================================="

# Papers under 100 words
echo
echo "Papers with ≤100 words (first 10):"
echo "-----------------------------------"
awk -F: '$2 <= 100 {print $1 " (" $2 " words)"}' "$PAPER_COUNTS" | head -10

# Papers 100-500 words
echo
echo "Papers with 100-500 words (first 10):"
echo "--------------------------------------"
awk -F: '$2 > 100 && $2 <= 500 {print $1 " (" $2 " words)"}' "$PAPER_COUNTS" | head -10

# Papers 500-1000 words
echo
echo "Papers with 500-1000 words (first 10):"
echo "---------------------------------------"
awk -F: '$2 > 500 && $2 <= 1000 {print $1 " (" $2 " words)"}' "$PAPER_COUNTS" | head -10

# Papers over 1000 words
echo
echo "Papers with >1000 words (first 10):"
echo "------------------------------------"
awk -F: '$2 > 1000 {print $1 " (" $2 " words)"}' "$PAPER_COUNTS" | head -10

echo
echo "Analysis complete!" 