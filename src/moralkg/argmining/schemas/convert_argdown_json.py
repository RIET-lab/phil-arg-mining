#!/usr/bin/env python3
"""
Converter script for transforming Argdown JSON format to custom argument mining JSON format.
TODO: Make the conversion process more modular, and integrate it with the syntax conversion script. Or, alternatively, add the json part of the other conversion script to this one to separate the syntax fixing from the argdown -> json pipeline.
"""

import json
import sys
from typing import Dict, List, Set, Any, Optional


def get_canonical_text(equivalence_class: Dict[str, Any]) -> str:
    """
    Extract the canonical text from an equivalence class.
    Prefers non-reference members with text.
    """
    members = equivalence_class.get('members', [])
    
    # First try to find a non-reference member with text
    for member in members:
        if not member.get('isReference', False) and member.get('text'):
            # Clean up the text (remove extra whitespace)
            text = member['text'].strip()
            # Remove "Main Thesis: " prefix if present
            if text.startswith('Main Thesis: '):
                text = text[13:]
            return text
    
    # Fallback to any member with text
    for member in members:
        if member.get('text'):
            text = member['text'].strip()
            if text.startswith('Main Thesis: '):
                text = text[13:]
            return text
    
    return ""


def find_root_nodes(statements: Dict[str, Any], relations: List[Dict[str, Any]]) -> Set[str]:
    """
    Identify root nodes (Major Claims) - nodes with no outgoing support relations.
    """
    # Get all nodes that are sources of support relations
    supported_nodes = set()
    for relation in relations:
        if relation.get('relationType') == 'support':
            supported_nodes.add(relation.get('from'))

    # Root nodes are those that are not supported by anything
    root_nodes = set()
    for title in statements.keys():
        if title not in supported_nodes:
            # Also check if it's marked as a top-level statement
            if statements[title].get('isUsedAsTopLevelStatement', False):
                root_nodes.add(title)

    return root_nodes


def convert_argdown_to_research_format(argdown_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Argdown JSON format to the research project's JSON format.
    """
    statements = argdown_json.get('statements', {})
    relations = argdown_json.get('relations', [])
    
    # Find root nodes (Major Claims)
    root_nodes = find_root_nodes(statements, relations)
    
    # Build ADUs dictionary
    adus = {}
    for title, equiv_class in statements.items():
        # Get the canonical text
        text = get_canonical_text(equiv_class)
        
        if text:  # Only include if we have text
            adu = {
                'type': 'Major Claim' if title in root_nodes else 'Claim',
                'text': text
            }
            
            # Check for optional fields in data
            data = equiv_class.get('data', {})
            
            # Add quote if present
            if 'quote' in data:
                adu['quote'] = data['quote']
            
            # Add isImplicit if true
            if data.get('isImplicit', False):
                adu['isImplicit'] = True
            
            adus[title] = adu
    
    # Build relations array
    converted_relations = []
    for relation in relations:
        # Only include relations where both source and target exist in our ADUs
        src = relation.get('from')
        tgt = relation.get('to')
        rel_type = relation.get('relationType')
        
        if src in adus and tgt in adus:
            # Convert "attack" to "attack" and everything else to "support"
            if rel_type in ['support', 'entails']:
                converted_type = 'support'
            elif rel_type in ['attack', 'contrary', 'contradictory', 'undercut']:
                converted_type = 'attack'
            else:
                converted_type = 'support'  # default
            
            converted_relations.append({
                'src': src,
                'tgt': tgt,
                'type': converted_type
            })
    
    return {
        'ADUs': adus,
        'relations': converted_relations
    }


def main():
    """
    Main function to handle file I/O and conversion.
    """
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python convert_argdown.py <input_file.json> [output_file.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.json', '_converted.json')
    
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            argdown_data = json.load(f)
        
        # Convert the data
        converted_data = convert_argdown_to_research_format(argdown_data)
        
        # Write the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"Successfully converted {input_file} to {output_file}")
        print(f"Extracted {len(converted_data['ADUs'])} ADUs:")
        
        major_claims = sum(1 for adu in converted_data['ADUs'].values() 
                          if adu['type'] == 'Major Claim')
        claims = len(converted_data['ADUs']) - major_claims
        
        print(f"  - {major_claims} Major Claim(s)")
        print(f"  - {claims} Claim(s)")
        print(f"Found {len(converted_data['relations'])} relations:")
        
        supports = sum(1 for rel in converted_data['relations'] 
                      if rel['type'] == 'support')
        attacks = len(converted_data['relations']) - supports
        
        print(f"  - {supports} support relation(s)")
        print(f"  - {attacks} attack relation(s)")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()