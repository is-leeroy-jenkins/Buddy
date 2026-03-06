"""Data utilities for Altex charts."""

from pathlib import Path

import pandas as pd

try:
    from streamlit import cache_data  # streamlit >= 1.18.0
except ImportError:
    from streamlit import experimental_memo as cache_data  # streamlit >= 0.89

# Get the directory where this file is located
DATA_DIR = Path(__file__).parent / "data"


@cache_data
def get_weather_data() -> pd.DataFrame:
    """Get sample weather data from Seattle.

    Returns:
        DataFrame with weather data including temperature, wind, etc.
    """
    return pd.read_csv(DATA_DIR / "weather.csv")


@cache_data
def get_stocks_data() -> pd.DataFrame:
    """Get sample stock price data.

    Returns:
        DataFrame with stock prices for different symbols over time.
    """
    return pd.read_csv(DATA_DIR / "stocks.csv").assign(
        date=lambda df: pd.to_datetime(df.date)
    )


@cache_data
def get_barley_data() -> pd.DataFrame:
    """Get sample barley yield data.

    Returns:
        DataFrame with barley yield data by variety and site.
    """
    return pd.read_json(DATA_DIR / "barley.json")


def get_random_data() -> pd.DataFrame:
    """Generate random sample data.

    Returns:
        DataFrame with random numerical data.
    """
    # Use pandas instead of numpy for random data
    import random

    random.seed(42)  # For reproducible "random" data
    data = {}

    for col in "abcdefg":
        # Generate 20 random values between -2 and 2
        data[col] = [random.uniform(-2, 2) for _ in range(20)]

    return pd.DataFrame(data).reset_index()
