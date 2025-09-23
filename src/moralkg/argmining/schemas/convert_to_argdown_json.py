#!/usr/bin/env python3
"""
Convert project-specific argument mining JSON back into an Argdown-like JSON format.

This is a best-effort reconstruction that creates a minimal Argdown JSON with:
  - "statements": equivalence-class objects with members containing the text
  - "relations": list of relation objects with relationType 'support' or 'attack'

The input format expected is the converted JSON produced by `convert_argdown_json.py`:
  {
    "ADUs": { "id": {"type": "Claim"|"Major Claim", "text": "...", ...}, ...},
    "relations": [{"src":"id","tgt":"id","type":"support"|"attack"}, ...]
  }

This script aims to be reversible enough for downstream tools that expect Argdown-style JSON.
"""

from typing import Dict, Any
import json
import sys
from pathlib import Path


def convert_to_argdown(converted_json: Dict[str, Any]) -> Dict[str, Any]:
    adus = converted_json.get('ADUs', {})
    relations = converted_json.get('relations', [])

    statements = {}
    out_relations = []

    # Create an equivalence-class for each ADU
    for title, adu in adus.items():
        # Sanitize text
        text = adu.get('text', '').strip()

        member = {
            'type': 'statement',
            'text': text,
            'title': title
        }

        # Add any optional fields into a data object
        data = {}
        if adu.get('isImplicit'):
            data['isImplicit'] = True
        if 'quote' in adu:
            data['quote'] = adu['quote']

        equiv = {
            'type': 'equivalence-class',
            'title': title,
            'members': [member],
        }
        if data:
            equiv['data'] = data

        # Mark top-level statements if ADU type indicates Major Claim
        if adu.get('type') == 'Major Claim':
            equiv['isUsedAsTopLevelStatement'] = True

        statements[title] = equiv

    # Reconstruct relations
    for rel in relations:
        src = rel.get('src')
        tgt = rel.get('tgt')
        rtype = rel.get('type', 'support')
        if src not in statements or tgt not in statements:
            # Skip relations referencing unknown nodes
            continue
        relation_obj = {
            'type': 'relation',
            'relationType': rtype if rtype in ('support', 'attack') else 'support',
            'from': src,
            'fromType': 'equivalence-class',
            'to': tgt,
            'toType': 'equivalence-class'
        }
        out_relations.append(relation_obj)

    # Minimal map and sections are omitted; include empty placeholders
    result = {
        'arguments': {},
        'statements': statements,
        'relations': out_relations,
        'map': {},
        'sections': [],
        'tags': {}
    }

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: convert_to_argdown_json.py <input_converted.json> [output_argdown.json]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_name(input_path.stem + '_to_argdown.json')

    if not input_path.exists():
        print(f"Error: input file {input_path} does not exist")
        sys.exit(1)

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    converted = convert_to_argdown(data)

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Wrote Argdown-like JSON to {output_path}")


if __name__ == '__main__':
    main()
