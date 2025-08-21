#!/bin/bash

# Interactive Sample Review Tool
#
# This script allows you to review each paper in your sample by:
# 1. Iterating through all papers in sample.csv
# 2. Showing the paper details and file location
# 3. Allowing you to open the file for review
# 4. Optionally launching the replacement script if you want to replace a paper
#
# Usage: ./sample-review.sh -n SAMPLE_SIZE [options]

set -e

# Default values
SAMPLE_SIZE=""
SAMPLE_DIR=""
SEED=42
START_FROM=""

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    echo "Usage: $0 -n SAMPLE_SIZE [options]"
    echo ""
    echo "Options:"
    echo "  -n, --sample-size SIZE    Sample size (required)"
    echo "  -d, --sample-dir DIR      Sample directory (default: data/annotations/samples/nSIZE)"
    echo "  -s, --seed SEED           Random seed for replacement script (default: 42)"
    echo "  -f, --start-from PAPER_ID Start from specific paper ID"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -n 100                 # Review sample of size 100"
    echo "  $0 -n 100 -f paper_123    # Start review from paper_123"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--sample-size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -d|--sample-dir)
            SAMPLE_DIR="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -f|--start-from)
            START_FROM="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SAMPLE_SIZE" ]]; then
    echo -e "${RED}Error: Sample size is required${NC}"
    usage
fi

# Set default sample directory if not provided
if [[ -z "$SAMPLE_DIR" ]]; then
    SAMPLE_DIR="data/annotations/samples/n${SAMPLE_SIZE}"
fi

# Check if sample directory exists
if [[ ! -d "$SAMPLE_DIR" ]]; then
    echo -e "${RED}Error: Sample directory does not exist: $SAMPLE_DIR${NC}"
    exit 1
fi

# Check if sample.csv exists
SAMPLE_FILE="$SAMPLE_DIR/sample.csv"
if [[ ! -f "$SAMPLE_FILE" ]]; then
    echo -e "${RED}Error: Sample file does not exist: $SAMPLE_FILE${NC}"
    exit 1
fi

# Function to get docling file path
get_docling_file() {
    local identifier="$1"
    local docling_dir="data/docling"
    
    if [[ -f "$docling_dir/$identifier.md" ]]; then
        echo "$docling_dir/$identifier.md"
    elif [[ -f "$docling_dir/$identifier.txt" ]]; then
        echo "$docling_dir/$identifier.txt"
    else
        echo ""
    fi
}

# Function to display paper info
display_paper_info() {
    local paper_id="$1"
    local index="$2"
    local total="$3"
    
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}Paper $index of $total${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo -e "${YELLOW}Paper ID:${NC} $paper_id"
    
    # Get docling file path
    local docling_file=$(get_docling_file "$paper_id")
    
    if [[ -n "$docling_file" ]]; then
        echo -e "${YELLOW}File Location:${NC} $docling_file"
        
        # Show file size and word count if file exists
        if [[ -f "$docling_file" ]]; then
            local file_size=$(du -h "$docling_file" | cut -f1)
            local word_count=$(wc -w < "$docling_file" 2>/dev/null || echo "N/A")
            echo -e "${YELLOW}File Size:${NC} $file_size"
            echo -e "${YELLOW}Word Count:${NC} $word_count"
        fi
    else
        echo -e "${RED}File Location: NOT FOUND${NC}"
        echo -e "${RED}Warning: Docling file does not exist for this paper${NC}"
    fi
}

# Function to show menu options
show_menu() {
    echo -e "\n${BLUE}What would you like to do?${NC}"
    echo "1) Open file for review"
    echo "2) Replace this paper"
    echo "3) Keep paper and continue to next"
    echo "4) Skip to specific paper ID"
    echo "5) Quit"
    echo -n "Choice [1-5]: "
}

# Function to open file
open_file() {
    local file_path="$1"
    
    if [[ ! -f "$file_path" ]]; then
        echo -e "${RED}Error: File does not exist: $file_path${NC}"
        return 1
    fi
    
    # Try different editors/viewers
    if command -v code >/dev/null 2>&1; then
        echo -e "${GREEN}Opening file in VS Code...${NC}"
        code "$file_path"
    elif command -v nano >/dev/null 2>&1; then
        echo -e "${GREEN}Opening file in nano...${NC}"
        nano "$file_path"
    elif command -v less >/dev/null 2>&1; then
        echo -e "${GREEN}Opening file in less...${NC}"
        less "$file_path"
    else
        echo -e "${YELLOW}No suitable editor found. File path: $file_path${NC}"
        echo -e "${YELLOW}You can copy this path and open it manually.${NC}"
    fi
}

# Function to run replacement script
run_replacement() {
    local paper_id="$1"
    
    echo -e "\n${GREEN}Launching replacement script for paper: $paper_id${NC}"
    echo -e "${YELLOW}Note: This will run the interactive replacement tool${NC}"
    
    # Check if sampling-replace.py exists
    local replace_script="data/scripts/annotations/samples/sampling-replace.py"
    if [[ ! -f "$replace_script" ]]; then
        echo -e "${RED}Error: Replacement script not found: $replace_script${NC}"
        return 1
    fi
    
    # Run the replacement script and capture exit code
    # Temporarily disable set -e to handle expected failures gracefully
    set +e
    python "$replace_script" -n "$SAMPLE_SIZE" -p "$paper_id" -s "$SEED"
    local exit_code=$?
    set -e
    
    # Handle different exit scenarios
    case $exit_code in
        0)
            echo -e "\n${GREEN}✓ Replacement completed successfully!${NC}"
            ;;
        1)
            echo -e "\n${YELLOW}⚠ No replacement candidates found for this paper${NC}"
            echo -e "${YELLOW}This paper is from a unique stratum with no alternatives available${NC}"
            echo -e "${CYAN}Continuing to next paper...${NC}"
            ;;
        *)
            echo -e "\n${RED}✗ Replacement script encountered an error (exit code: $exit_code)${NC}"
            echo -e "${RED}You may want to check the logs for details${NC}"
            ;;
    esac
    
    echo -e "\n${BLUE}Press Enter to continue...${NC}"
    read -r
}

# Main execution
echo -e "${GREEN}Sample Review Tool${NC}"
echo -e "${GREEN}==================${NC}"
echo -e "Sample Directory: $SAMPLE_DIR"
echo -e "Sample Size: $SAMPLE_SIZE"
echo -e "Random Seed: $SEED"

# Read paper IDs from sample.csv
echo -e "\n${BLUE}Loading papers from sample...${NC}"

# Get paper IDs (skip header)
mapfile -t paper_ids < <(tail -n +2 "$SAMPLE_FILE" | cut -d',' -f1)

total_papers=${#paper_ids[@]}
echo -e "${GREEN}Found $total_papers papers in sample${NC}"

# Find starting index
start_index=0
if [[ -n "$START_FROM" ]]; then
    for i in "${!paper_ids[@]}"; do
        if [[ "${paper_ids[$i]}" == "$START_FROM" ]]; then
            start_index=$i
            echo -e "${YELLOW}Starting from paper: $START_FROM (position $((i+1)))${NC}"
            break
        fi
    done
    
    if [[ $start_index -eq 0 && "${paper_ids[0]}" != "$START_FROM" ]]; then
        echo -e "${RED}Warning: Paper $START_FROM not found in sample. Starting from beginning.${NC}"
    fi
fi

# Main review loop
current_index=$start_index
while [[ $current_index -lt $total_papers ]]; do
    paper_id="${paper_ids[$current_index]}"
    
    display_paper_info "$paper_id" $((current_index + 1)) "$total_papers"
    
    while true; do
        show_menu
        read -r choice
        
        case $choice in
            1)
                docling_file=$(get_docling_file "$paper_id")
                if [[ -n "$docling_file" ]]; then
                    open_file "$docling_file"
                else
                    echo -e "${RED}No file to open for this paper${NC}"
                fi
                ;;
            2)
                run_replacement "$paper_id"
                # After replacement, continue to next paper
                break
                ;;
            3)
                echo -e "${GREEN}Keeping paper and continuing...${NC}"
                break
                ;;
            4)
                echo -n "Enter paper ID to skip to: "
                read -r target_paper
                found=false
                for i in "${!paper_ids[@]}"; do
                    if [[ "${paper_ids[$i]}" == "$target_paper" ]]; then
                        current_index=$i
                        found=true
                        break
                    fi
                done
                if [[ $found == true ]]; then
                    echo -e "${GREEN}Skipping to paper: $target_paper${NC}"
                    break
                else
                    echo -e "${RED}Paper $target_paper not found in sample${NC}"
                fi
                ;;
            5)
                echo -e "${YELLOW}Exiting sample review...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Please enter 1-5.${NC}"
                ;;
        esac
    done
    
    current_index=$((current_index + 1))
done

echo -e "\n${GREEN}Sample review completed!${NC}"
echo -e "${GREEN}All papers have been reviewed.${NC}" 