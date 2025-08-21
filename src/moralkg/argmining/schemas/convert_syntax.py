#!/usr/bin/env python3
"""
Converts custom argdown syntax to standard argdown syntax and then to JSON.
In the custom syntax:
  - '+' indicates an explicit claim
  - '-' indicates an implicit claim
Both are converted to support relations (+) with implicit claims getting {isImplicit: True} metadata.
"""

import re
import sys
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Tuple


def convert_custom_syntax(lines: List[str]) -> List[str]:
    """
    Convert custom argdown syntax to standard argdown syntax.
    Changes '-' to '+' and adds {isImplicit: True} metadata where needed.
    """
    converted_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts with a relation symbol (after whitespace)
        match = re.match(r'^(\s*)([-+])\s+(.+)$', line)
        
        if match:
            indent = match.group(1)
            symbol = match.group(2)
            content = match.group(3)
            
            # Convert to support relation
            converted_line = f"{indent}+ {content}"
            converted_lines.append(converted_line)
            
            # If it was implicit (originally '-'), add metadata
            if symbol == '-':
                # Look ahead to see if the next line is already a metadata line or another relation
                next_line_is_metadata = False
                next_line_is_relation = False
                
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # Check if next line is metadata (starts with more indent and '{')
                    if next_line.strip().startswith('{'):
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent > len(indent):
                            next_line_is_metadata = True
                    # Check if next line is another relation at same or less indentation
                    next_match = re.match(r'^(\s*)([-+])\s+', next_line)
                    if next_match:
                        next_indent = len(next_match.group(1))
                        if next_indent <= len(indent):
                            next_line_is_relation = True
                
                # Add metadata line with same indentation as the statement
                # Only add if there isn't already metadata on the next line
                if not next_line_is_metadata:
                    metadata_indent = indent + "    "  # Add one more level of indentation
                    converted_lines.append(f"{metadata_indent}{{isImplicit: true}}")
        else:
            # Not a relation line, keep as is
            converted_lines.append(line)
        
        i += 1
    
    return converted_lines


def process_argdown_file(input_file: str, output_argdown: str = None, output_json: str = None) -> None:
    """
    Process a custom argdown file:
    1. Convert custom syntax to standard argdown
    2. Save the converted argdown
    3. Use argdown CLI to convert to JSON
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Remove trailing newlines but preserve the line structure
    lines = [line.rstrip('\n') for line in lines]
    
    # Convert the syntax
    converted_lines = convert_custom_syntax(lines)
    
    # Determine output filenames
    input_path = Path(input_file)
    base_name = input_path.stem
    
    if output_argdown is None:
        output_argdown = input_path.parent / f"{base_name}_converted.argdown"
    else:
        output_argdown = Path(output_argdown)
    
    if output_json is None:
        output_json = input_path.parent / f"{base_name}.json"
    else:
        output_json = Path(output_json)
    
    # Write the converted argdown
    with open(output_argdown, 'w', encoding='utf-8') as f:
        f.write('\n'.join(converted_lines))
    
    print(f"Converted argdown saved to: {output_argdown}")
    
    # Check if argdown CLI is available (try local install first, then global)
    argdown_cmd = None
    
    # Try to find argdown CLI
    # 1. Try local installation with npx
    try:
        result = subprocess.run(['npx', 'argdown', '--version'], 
                              capture_output=True, text=True, check=True)
        argdown_cmd = ['npx', 'argdown']
        print(f"Using local argdown CLI version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 2. Try global installation
        try:
            result = subprocess.run(['argdown', '--version'], 
                                  capture_output=True, text=True, check=True)
            argdown_cmd = ['argdown']
            print(f"Using global argdown CLI version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # 3. Try direct path to local node_modules
            local_argdown = Path('./node_modules/.bin/argdown')
            if local_argdown.exists():
                argdown_cmd = [str(local_argdown)]
                print(f"Using argdown CLI from node_modules")
            else:
                print("Error: argdown CLI not found. Please install it with:")
                print("  npm install @argdown/cli  (for local installation)")
                print("  or")
                print("  npm install -g @argdown/cli  (for global installation)")
                print("\nConverted argdown file has been saved, but JSON conversion skipped.")
                return
    
    # Run argdown CLI to convert to JSON
    try:
        # Create output directory if it doesn't exist
        output_json.parent.mkdir(parents=True, exist_ok=True)
        
        # Run argdown json command
        cmd = argdown_cmd + ['json', str(output_argdown), str(output_json.parent)]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # The argdown CLI creates a file with the same name but .json extension
        expected_json = output_json.parent / f"{output_argdown.stem}.json"
        
        # If the output file is different from what we want, rename it
        if expected_json != output_json and expected_json.exists():
            expected_json.rename(output_json)
        
        print(f"JSON file saved to: {output_json}")
        
        # Print any output from argdown CLI
        if result.stdout:
            print("Argdown CLI output:", result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"Error running argdown CLI: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        print("\nConverted argdown file has been saved, but JSON conversion failed.")


def main():
    """
    Main function to handle command line arguments.
    """
    if len(sys.argv) < 2:
        print("Usage: python convert_syntax.py <input.argdown> [output.argdown] [output.json]")
        print("\nThis script converts custom argdown syntax where:")
        print("  '+' = explicit claim")
        print("  '-' = implicit claim")
        print("\nTo standard argdown syntax with YAML metadata for implicit claims.")
        print("\nThen uses argdown CLI to convert to JSON format.")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_argdown = sys.argv[2] if len(sys.argv) > 2 else None
    output_json = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    try:
        process_argdown_file(input_file, output_argdown, output_json)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()