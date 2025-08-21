# lang-distribution.py

# Assumes you're running from `moralkg/data/scripts/figures`

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pycountry 
import argparse
import glob
import os


def get_most_recent_metadata_file():
    """Find the most recent metadata CSV file in data/metadata/ directory."""
    metadata_pattern = '../../metadata/metadata-*.csv'
    metadata_files = glob.glob(metadata_pattern)
    
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found matching pattern: {metadata_pattern}")
    
    # Sort by modification time, most recent first
    most_recent = max(metadata_files, key=os.path.getmtime)
    return most_recent


def main():
    parser = argparse.ArgumentParser(
        description='Generate language distribution plot from metadata.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'filename', 
        nargs='?',
        help='Name of the metadata CSV file in data/metadata/ directory. '
             'If not provided, uses the most recent metadata file.'
    )
    
    parser.add_argument(
        '--file', '-f',
        dest='filename_flag',
        help='Alternative way to specify the metadata filename using flag.'
    )
    
    args = parser.parse_args()
    
    # Determine which filename to use
    if args.filename_flag:
        filename = args.filename_flag
    elif args.filename:
        filename = args.filename
    else:
        filename = None
    
    # Get the full path to the metadata file
    if filename:
        if not filename.startswith('../../metadata/'):
            metadata_path = f'../../metadata/{filename}'
        else:
            metadata_path = filename
    else:
        metadata_path = get_most_recent_metadata_file()
        print(f"Using most recent metadata file: {metadata_path}")
    
    # Check if file exists
    if not os.path.exists(metadata_path):
        print(f"Error: File not found: {metadata_path}")
        return 1

    # Load CSV
    df = pd.read_csv(metadata_path)

    # Compute top 10; lump the rest into “Other”
    counts      = df['language'].value_counts()
    top_counts  = counts.iloc[:10].copy()
    top_counts['Other'] = counts.iloc[10:].sum()

    # Code → Name map from ISO 639-1
    lang_map = {l.alpha_2: l.name for l in pycountry.languages if hasattr(l, 'alpha_2')}

    # Turn into a DataFrame and translate codes
    plot_df = top_counts.reset_index()
    plot_df.columns = ['code', 'count']
    plot_df['language'] = plot_df['code'].map(lang_map).fillna(plot_df['code'])

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='language', y='count')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Language')
    plt.ylabel('Count')
    plt.title('Top 10 Languages + Others')
    plt.tight_layout()
    plt.savefig('../../figures/language-distribution.png')
    
    print(f"Language distribution plot saved to: moralkg/data/figures/language-distribution.png")
    return 0


if __name__ == '__main__':
    main()
