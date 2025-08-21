from pathlib import Path
import pandas as pd
import ftfy
from moralkg.logging import get_logger


def _transform_type(value):
    if pd.isna(value) or not isinstance(value, str):
        return value
    return value.split('/')[-1]


def reformat_combined_metadata(input_csv: str, output_csv: str) -> None:
    """Reorder columns, normalize type, and fix unicode text in authors/title."""
    df = pd.read_csv(Path(input_csv))
    if 'type' in df.columns:
        df['type'] = df['type'].apply(_transform_type)
    if 'authors' in df.columns:
        df['authors'] = df['authors'].apply(lambda x: ftfy.fix_text(x) if pd.notna(x) else x)
    if 'title' in df.columns:
        df['title'] = df['title'].apply(lambda x: ftfy.fix_text(x) if pd.notna(x) else x)

    desired_order = [
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
        'identifier-url',
    ]

    # Keep any missing columns as-is at the end, only reorder those present
    order = [c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]
    df_reordered = df[order]
    df_reordered.to_csv(Path(output_csv), index=False, quoting=0)
    get_logger(__name__).info("Reformatted metadata saved to %s", output_csv)


