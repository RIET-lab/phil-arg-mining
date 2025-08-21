#!/usr/bin/env python3
"""
Test the new year parsing utility vs the old approach.
"""

import pandas as pd
import sys
sys.path.append('data/scripts/annotations/samples')
from utils import parse_year, get_year_parsing_stats

def main():
    print("Loading metadata...")
    df = pd.read_csv('data/metadata/2025-07-09-en-combined-metadata.csv')
    
    print(f"Total rows: {len(df)}")
    
    # Test old approach (pd.to_numeric)
    print("\n=== OLD APPROACH (pd.to_numeric) ===")
    old_parsed = pd.to_numeric(df['year'], errors='coerce')
    old_success = old_parsed.notna().sum()
    old_lost = df['year'].notna().sum() - old_success
    print(f"Successfully parsed: {old_success}")
    print(f"Lost entries: {old_lost}")
    print(f"Success rate: {old_success / df['year'].notna().sum():.1%}")
    
    # Test new approach
    print("\n=== NEW APPROACH (utils.parse_year) ===")
    stats = get_year_parsing_stats(df)
    print(f"Successfully parsed: {stats['successfully_parsed']}")
    print(f"Lost entries: {stats['lost_entries']}")
    print(f"Success rate: {stats['parsing_success_rate']:.1%}")
    
    improvement = stats['successfully_parsed'] - old_success
    print(f"\n=== IMPROVEMENT ===")
    print(f"Additional entries recovered: {improvement}")
    print(f"Improvement: {improvement / df['year'].notna().sum():.1%}")
    
    # Test some specific examples
    print("\n=== EXAMPLE PARSING ===")
    test_cases = [
        'manuscript', 'forthcoming', 'unknown',
        '2019', '1995', 'Oct 28, 2019', 'Jan 2019',
        '1st ed. 2016', '2010 (Hardcover)',
        'unJuly 2011known', 'unMay, 2015known',
        '212', '1998/99', '201?'
    ]
    
    for case in test_cases:
        result = parse_year(case)
        print(f"'{case}' -> {result}")

if __name__ == '__main__':
    main() 