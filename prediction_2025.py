"""
This module trains fare prediction models per airline route using historical data (2018–2024),
and generates fare forecasts for each route for all quarters of 2025.

It uses linear regression to detect fare trends per route over time.
"""
import os
import pandas as pd
from sklearn.linear_model import LinearRegression


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame from the CSV.
    """
    return pd.read_csv(file_path)


def add_route_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'route' column by combining airport_1 and airport_2.
    """
    df['route'] = df['airport_1'] + " - " + df['airport_2']
    return df


def train_and_predict_fares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trains linear models per route and predicts fare for 2025 Q1 to Q4.
    Returns a DataFrame of predicted rows.
    """
    model_data = []

    for _, group in df.groupby('route'):
        if group['Year'].nunique() < 3:
            continue

        group = group.sort_values(by=['Year', 'quarter'])
        x = group['Year'] + (group['quarter'] - 1) / 4
        x = x.values.reshape(-1, 1)
        y = group['fare']

        model = LinearRegression()
        model.fit(x, y)

        for q in range(1, 5):
            decimal_time = 2025 + (q - 1) / 4
            predicted_fare = round(model.predict([[decimal_time]])[0], 2)

            sample_row = group.iloc[-1].copy()
            sample_row['Year'] = 2025
            sample_row['quarter'] = q
            sample_row['fare'] = predicted_fare
            model_data.append(sample_row)

    return pd.DataFrame(model_data)


def sample_predictions(df: pd.DataFrame, samples_per_quarter: int = 1000) -> pd.DataFrame:
    """
    Ensures each quarter has a fixed number of samples (default 1000).
    """
    sampled_df = df.groupby('quarter').apply(
        lambda x: x.sample(n=samples_per_quarter, random_state=42) if len(x) > samples_per_quarter else x
    )
    return sampled_df.reset_index(drop=True)


def clean_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up temporary columns.
    """
    columns_to_drop = ['route']
    return df.drop(columns=[col for col in columns_to_drop if col in df.columns])


def remove_negative_fares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows where the predicted fare is negative.
    """
    return df[df['fare'] >= 0]


def save_predictions_to_csv(df: pd.DataFrame, filename: str) -> None:
    """
    Saves the DataFrame to a CSV file.
    """
    df.to_csv(filename, index=False)


def test_prediction_output() -> None:
    """
    Tests that the 2025 fare prediction pipeline preserves all relevant columns
    and produces valid values for year, quarter, and fare.

    You can simply use this to test the file.
    """
    input_csv = "USA_Filtered_Airline_2018-2024.csv"
    output_csv = "Predicted_Airlines_2025.csv"

    original_df = load_csv_data(input_csv)
    original_df = add_route_column(original_df)
    predicted_df = train_and_predict_fares(original_df)
    predicted_df = sample_predictions(predicted_df)
    predicted_df = remove_negative_fares(predicted_df)
    predicted_df = clean_predictions(predicted_df)
    save_predictions_to_csv(predicted_df, output_csv)

    assert os.path.exists(output_csv), "Output CSV file was not created."

    # Reload output for testing
    output_df = pd.read_csv(output_csv)

    # Ensure all years are 2025
    assert output_df['Year'].eq(2025).all(), "Not all Year values are 2025."

    # Ensure all quarters are between 1 and 4
    assert output_df['quarter'].between(1, 4).all(), "Quarter values out of range."

    # Ensure all fares are non-negative
    assert (output_df['fare'] >= 0).all(), "Negative fare values found."

    # Columns in output should match input (except for route which is dropped)
    expected_columns = set(original_df.columns) - {'route'}
    assert expected_columns.issubset(set(output_df.columns)), \
        f"Missing columns in prediction output: {expected_columns - set(output_df.columns)}"


if __name__ == '__main__':
    # When you are ready to check your work with python_ta, uncomment the following lines.
    # (In PyCharm, select the lines below and press Ctrl/Cmd + / to toggle comments.)
    # You can use "Run file in Python Console" to run both pytest and PythonTA,
    # and then also test your methods manually in the console.
    import python_ta
    python_ta.check_all(config={
        'extra-imports': ['pandas', 'sklearn.linear_model', 'os'],  # the names (strs) of imported modules
        'allowed-io': [],  # the names (strs) of functions that call print/open/input
        'max-line-length': 120
    })
