#!/usr/bin/env python3
"""
Analyze year column patterns in the metadata file.
"""

import pandas as pd
import re
from collections import Counter

def main():
    print("Loading metadata...")
    df = pd.read_csv('data/metadata/2025-07-09-en-combined-metadata.csv')
    
    print(f"Total rows: {len(df)}")
    print(f"Year column name: {df.columns[3]}")  # Should be 'year'
    
    year_col = df['year']
    
    # Count non-null values
    non_null_years = year_col.dropna()
    print(f"Non-null years: {len(non_null_years)}")
    
    # Convert to string to analyze patterns
    year_strings = non_null_years.astype(str)
    
    # Check for pure numeric years (4 digits)
    pure_numeric = year_strings.str.match(r'^\d{4}$')
    numeric_count = pure_numeric.sum()
    print(f"Pure 4-digit numeric years: {numeric_count}")
    
    # Find non-numeric patterns
    non_numeric = year_strings[~pure_numeric]
    print(f"Non-numeric year entries: {len(non_numeric)}")
    
    if len(non_numeric) > 0:
        print("\nSample non-numeric year patterns:")
        patterns = Counter(non_numeric)
        for pattern, count in patterns.most_common(20):
            print(f"  '{pattern}': {count} occurrences")
        
        print(f"\nTotal unique non-numeric patterns: {len(patterns)}")
        
        # Check for common patterns like "Jan 2019", "2019-2020", etc.
        month_year_pattern = non_numeric.str.match(r'.*\d{4}.*')
        contains_year = month_year_pattern.sum()
        print(f"Non-numeric entries that contain a 4-digit year: {contains_year}")
        
        if contains_year > 0:
            print("\nSample entries containing years:")
            year_containing = non_numeric[month_year_pattern]
            for sample in year_containing.head(10):
                print(f"  '{sample}'")

if __name__ == '__main__':
    main() 