# written to reformat 2025-07-09-combined-metadata.csv to be a different column
# order and to not use quotes around the values

# also deals with unicode issues across authors and titles

import pandas as pd
from pathlib import Path
import rootutils
import ftfy


def transform_type(type_value):
    if pd.isna(type_value) or not isinstance(type_value, str):
        return type_value
    return type_value.split('/')[-1]


def main():
    root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True)
    input_file = root / 'data' / 'metadata' / '2025-07-09-en-combined-metadata.csv'
    output_file = root / 'data' / 'metadata' / '2025-07-09-en-combined-metadata-reformatted.csv'
        
    df = pd.read_csv(input_file)
    df['type'] = df['type'].apply(transform_type)
    
    df['authors'] = df['authors'].apply(lambda x: ftfy.fix_text(x) if pd.notna(x) else x)
    df['title'] = df['title'].apply(lambda x: ftfy.fix_text(x) if pd.notna(x) else x)
    
    reorder = [
        'identifier',
        'type', 
        'language',
        'year',
        'authors',
        'title',
        'subjects',
        'category_names',
        'category_ids',
        'num_categories',
        'identifier-url'
    ]
    
    df_reordered = df[reorder]
    df_reordered.to_csv(output_file, index=False, quoting=0)
    
if __name__ == '__main__':
    main()