#!/usr/bin/env python3
"""
Utility functions for preprocessing the year column.
"""

import re
import pandas as pd
from typing import Union, Optional


def parse_year(year_value: Union[str, int, float]) -> Optional[int]:
    """
    Parse year from various formats, returning a 4-digit integer year or None.

    Handles cases like:
    - Pure numeric: 2019, "2019"
    - Date strings: "Oct 28, 2019", "Jan 2019"
    - Edition strings: "1st ed. 2016"
    - Corrupted entries: "unJuly 2011known"
    - Special cases: "manuscript", "forthcoming", "unknown" -> None

    Args:
        year_value: The year value to parse

    Returns:
        4-digit integer year if parseable, None otherwise
    """
    if pd.isna(year_value):
        return None

    # Convert to string for processing
    year_str = str(year_value).strip()

    # Handle obvious non-year cases
    non_year_keywords = {
        "manuscript",
        "forthcoming",
        "unknown",
        "not applicable",
        "n/a",
        "na",
        "tbd",
        "pending",
    }
    if year_str.lower() in non_year_keywords:
        return None

    # First try: direct 4-digit number
    if re.match(r"^\d{4}$", year_str):
        year = int(year_str)
        # Sanity check for reasonable year range
        if 1000 <= year <= 2050:
            return year
        return None

    # Extract 4-digit year from anywhere in the string
    # First try with word boundaries
    year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", year_str)
    if not year_matches:
        # If no word boundary matches, try without boundaries (for corrupted entries)
        year_matches = re.findall(r"(19\d{2}|20\d{2})", year_str)

    if year_matches:
        # Take the first reasonable year found
        year = int(year_matches[0])
        if 1000 <= year <= 2050:
            return year

    # Handle 2-digit years with some context
    # e.g., "98" -> 1998, "05" -> 2005
    two_digit_match = re.search(r"\b(\d{2})\b", year_str)
    if two_digit_match:
        two_digit = int(two_digit_match.group(1))
        # Assume 00-30 is 2000s, 31-99 is 1900s
        if 0 <= two_digit <= 30:
            return 2000 + two_digit
        elif 31 <= two_digit <= 99:
            return 1900 + two_digit

    return None


def extract_year_column(df: pd.DataFrame, year_column: str = "year") -> pd.Series:
    """
    Apply robust year parsing to a DataFrame column.

    Args:
        df: DataFrame containing the year column
        year_column: Name of the column containing year data

    Returns:
        Series of parsed integer years with NaN for unparseable entries
    """
    if year_column not in df.columns:
        raise KeyError(f"Column '{year_column}' not found in DataFrame")

    # Explicitly cast to Series to satisfy type checker
    return pd.Series(df[year_column].apply(parse_year))


def get_year_parsing_stats(df: pd.DataFrame, year_column: str = "year") -> dict:
    """
    Get statistics about year parsing success.

    Args:
        df: DataFrame containing the year column
        year_column: Name of the column containing year data

    Returns:
        Dictionary with parsing statistics
    """
    original_col = df[year_column]
    parsed_col = extract_year_column(df, year_column)

    stats = {
        "total_rows": len(df),
        "non_null_original": original_col.notna().sum(),
        "successfully_parsed": parsed_col.notna().sum(),
        "parsing_success_rate": (
            parsed_col.notna().sum() / original_col.notna().sum()
            if original_col.notna().sum() > 0
            else 0
        ),
        "lost_entries": original_col.notna().sum() - parsed_col.notna().sum(),
    }

    return stats
