#!/usr/bin/env python3
"""
Quick test of Gower multiprocessing with small sample.
"""

import gower_multiprocessing as gower
import numpy as np
import pandas as pd
import rootutils
import time
from utils import extract_year_column

def main():        
    # Set up paths
    root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True)
    input_file = root / 'data' / 'metadata' / '2025-07-09-en-combined-metadata.csv'
    
    print("Testing Gower multiprocessing with small sample...")
    
    # Load and sample data
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    
    # Select subset of columns and sample small dataset
    cols = ['title', 'authors', 'year', 'num_categories', 'category_names']
    df_subset = df[cols].dropna().copy()
    df_subset['year'] = extract_year_column(df_subset, 'year')
    df_subset = df_subset.dropna()
    
    # Test with different sizes
    test_sizes = [1000, 5000, 10000]
    
    for size in test_sizes:
        if len(df_subset) >= size:
            sample_df = df_subset.sample(n=size, random_state=42)
            print(f"\nTesting with {size} records...")
            
            start_time = time.time()
            distance_matrix = gower.gower_matrix(sample_df)
            end_time = time.time()
            
            print(f"  Time: {end_time - start_time:.2f} seconds")
            print(f"  Matrix shape: {distance_matrix.shape}")
            print(f"  Records per second: {size / (end_time - start_time):.0f}")
            
            # Test if we're getting reasonable speedup
            if size >= 5000:
                expected_time_single = (size ** 2) / (10000 ** 2) * 120  # Rough estimate
                speedup = expected_time_single / (end_time - start_time)
                print(f"  Estimated speedup: {speedup:.1f}x")

if __name__ == '__main__':
    main() 