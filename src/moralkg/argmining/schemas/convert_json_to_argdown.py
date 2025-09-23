#!/usr/bin/env python3
"""
Convert the project's custom argument-mining JSON or an Argdown JSON back into Argdown text (.argdown).

Usage:
  python convert_json_to_argdown.py input.json [output.argdown]

The script supports both formats:
 - Custom format: top-level keys 'ADUs' and 'relations'
 - Argdown JSON: top-level keys 'statements' (equivalence-classes) and 'relations'

The output is a best-effort nested Argdown text that prints Major Claims as top-level
blocks and nests supporters/attackers underneath using '+' for support and '-' for attack.
If an ADU has `isImplicit: true`, a metadata line `{isImplicit: True}` is added under the statement.
"""

from pathlib import Path
from typing import Dict, Any, List, Set
import json
import sys


def load_input(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    # Detect custom format
    if 'ADUs' in data and 'relations' in data:
        return data

    # Detect argdown-json 'statements' format
    if 'statements' in data and 'relations' in data:
        # Convert to ADUs/relations style for easier handling
        adus = {}
        for title, equiv in data['statements'].items():
            # get canonical text from first member
            members = equiv.get('members', [])
            text = ''
            if members:
                text = members[0].get('text', '').strip()
            adu = {'type': 'Claim', 'text': text}
            # carry over top-level flag
            if equiv.get('isUsedAsTopLevelStatement'):
                adu['type'] = 'Major Claim'
            # carry over other data
            if 'data' in equiv and equiv['data'].get('isImplicit'):
                adu['isImplicit'] = True
            adus[title] = adu

        return {'ADUs': adus, 'relations': data.get('relations', [])}

    raise ValueError('Unrecognized input JSON format')


def build_relation_index(relations: List[Dict[str, Any]]):
    # Map target -> list of (src, type)
    idx = {}
    for r in relations:
        # support may be under 'type' or 'relationType' in different formats
        if 'relationType' in r:
            rtype = r.get('relationType')
        else:
            rtype = r.get('type')
        # Normalize to 'support' or 'attack'
        if rtype not in ('support', 'attack'):
            rtype = 'support'

        src = r.get('from') or r.get('src')
        tgt = r.get('to') or r.get('tgt')
        if not src or not tgt:
            continue
        # preserve relation order by appending in input order
        idx.setdefault(tgt, []).append((src, rtype))
    return idx


def write_argdown(adus: Dict[str, Dict[str, Any]], relations: List[Dict[str, Any]], out_path: Path):
    rel_idx = build_relation_index(relations)

    def write_block(f, title: str, level: int, visited: Set[str]):
        # Depth-first traversal following relation list order.
        # Only print the header for top-level blocks (level == 0). For child nodes,
        # print them as nested bullet lines and recurse to print their children.
        if title in visited:
            return
        visited.add(title)
        adu = adus.get(title, {})
        indent = '\t' * level

        # Title/header only for top-level blocks
        if level == 0:
            text = adu.get('text', '').strip()
            f.write(f"[{title}]: {text}\n")
            if adu.get('isImplicit'):
                f.write(f"{indent}{{isImplicit: True}}\n")

        # Children in the order found in relations
        children = rel_idx.get(title, [])
        for src, rtype in children:
            marker = '+' if rtype == 'support' else '-'
            src_adu = adus.get(src, {})
            src_text = src_adu.get('text', '').strip()
            # Print the child as a bullet line under the current block
            f.write(f"\t{marker} {src_text}\n")
            if src_adu.get('isImplicit'):
                f.write(f"\t\t{{isImplicit: True}}\n")
            # Recurse to print the grandchildren under the child (deeper indent),
            # but do not reprint the child's header.
            # Note: we still pass the child's title so recursion can find its children.
            # To avoid duplicate headers, the recursive call will not print header because level>0.
            write_block(f, src, level + 1, visited)

    # Write top-level blocks: prefer ADUs of type 'Major Claim'
    with out_path.open('w', encoding='utf-8') as f:
        majors = [k for k, v in adus.items() if v.get('type') == 'Major Claim']
        # If no majors, pick nodes that are not source of any relation (roots)
        if not majors:
            # targets that have no outgoing relation? fallback: all nodes
            majors = list(adus.keys())

        visited = set()
        for m in majors:
            write_block(f, m, 0, visited)
            f.write('\n\n')


def main():
    if len(sys.argv) < 2:
        print('Usage: convert_json_to_argdown.py <input.json> [output.argdown]')
        sys.exit(1)

    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_suffix('.argdown')

    if not inp.exists():
        print(f'Input file {inp} not found')
        sys.exit(1)

    data = load_input(inp)
    adus = data.get('ADUs', {})
    relations = data.get('relations', [])

    write_argdown(adus, relations, out)
    print(f'Wrote Argdown to {out}')


if __name__ == '__main__':
    main()
