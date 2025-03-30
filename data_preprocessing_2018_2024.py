"""
This module provides data preprocessing functions for airline flight data,
including geocode extraction, cleaning, validation, and CSV exporting.
"""
import os
import re
import pandas as pd


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def categorize_period(year: int) -> str:
    """
    Categorizes a given year into a pandemic-related period.
    """
    if 2018 <= year <= 2020:
        return "Pre-Pandemic"
    elif 2021 <= year <= 2022:
        return "During-Pandemic"
    elif 2023 <= year <= 2024:
        return "Post-Pandemic"
    else:
        return "Other"


def add_period_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'period' column based on the 'Year' column.
    """
    df['period'] = df['Year'].apply(categorize_period)
    return df


def filter_and_sort_by_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows labeled as 'Other' and sorts by Year and Quarter.
    """
    df = df[df['period'] != 'Other']
    return df.sort_values(by=['Year', 'quarter']).reset_index(drop=True)


def build_geocode_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a lookup for known geocoded city pairs.
    """
    return (
        df.dropna(subset=['Geocoded_City1', 'Geocoded_City2'])
        .drop_duplicates(subset=['city1', 'city2'])
        .set_index(['city1', 'city2'])[['Geocoded_City1', 'Geocoded_City2']]
    )


def fill_missing_geocodes(df: pd.DataFrame, geo_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing geocodes using the lookup table.
    """
    def fill_geocodes(row: pd.Series) -> pd.Series:
        if pd.isna(row['Geocoded_City1']) or pd.isna(row['Geocoded_City2']):
            key = (row['city1'], row['city2'])
            if key in geo_lookup.index:
                if pd.isna(row['Geocoded_City1']):
                    row['Geocoded_City1'] = geo_lookup.loc[key, 'Geocoded_City1']
                if pd.isna(row['Geocoded_City2']):
                    row['Geocoded_City2'] = geo_lookup.loc[key, 'Geocoded_City2']
        return row
    return df.apply(fill_geocodes, axis=1)


def extract_tuple(geo_str: str) -> str | None:
    """
    Converts raw geocode strings into (lat, lon) format.
    """
    match = re.findall(r'-?\d+\.\d+', str(geo_str))
    return f"({match[0]}, {match[1]})" if len(match) >= 2 else None


def clean_geocode_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes Geocoded_City1 and Geocoded_City2 columns.
    """
    for col in ['Geocoded_City1', 'Geocoded_City2']:
        if col in df.columns:
            df[col] = df[col].apply(extract_tuple)
        else:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    return df


def is_valid_geocode(val: str) -> bool:
    """
    Validates that a geocode is in (lat, lon) format.
    """
    return isinstance(val, str) and "(" in val and "," in val and ")" in val


def filter_valid_geocodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keeps only rows with valid geocodes.
    """
    return df[
        df['Geocoded_City1'].apply(is_valid_geocode)
        & df['Geocoded_City2'].apply(is_valid_geocode)
    ].reset_index(drop=True)


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns not needed for modeling.
    """
    columns_to_drop = [
        'carrier_lg', 'large_ms', 'fare_lg',
        'carrier_low', 'lf_ms', 'fare_low',
        'passengers', 'tbl'
    ]
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns])


def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the cleaned DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)


def run_preprocessing_pipeline(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Complete preprocessing pipeline from raw to cleaned dataset.

    Returns the cleaned DataFrame after saving it to a CSV.


    """
    df = load_csv_data(input_csv)
    df = add_period_column(df)
    df = filter_and_sort_by_period(df)
    geo_lookup = build_geocode_lookup(df)
    df = fill_missing_geocodes(df, geo_lookup)
    df = clean_geocode_strings(df)
    df = filter_valid_geocodes(df)
    df = drop_unnecessary_columns(df)
    save_cleaned_data(df, output_csv)
    return df


def test_output_file_creation() -> None:
    """
    Tests that the preprocessing pipeline generates the expected output CSV file
    and that its content matches the returned DataFrame.

    You can simply use this to test the file.
    """
    input_csv = "US Airline Flight Routes and Fares 1993-2024.csv"
    output_csv = "USA_Filtered_Airline_2018-2024.csv"

    cleaned_df = run_preprocessing_pipeline(input_csv, output_csv)

    assert os.path.exists(output_csv), f"{output_csv} was not created."

    output_df = pd.read_csv(output_csv)
    assert cleaned_df.shape == output_df.shape
    assert list(cleaned_df.columns) == list(output_df.columns)


if __name__ == '__main__':
    # When you are ready to check your work with python_ta, uncomment the following lines.
    # (In PyCharm, select the lines below and press Ctrl/Cmd + / to toggle comments.)
    # You can use "Run file in Python Console" to run both pytest and PythonTA,
    # and then also test your methods manually in the console.
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['pandas', 're', 'os'],  # the names (strs) of imported modules
        'allowed-io': [],  # the names (strs) of functions that call print/open/input
        'max-line-length': 120
    })
