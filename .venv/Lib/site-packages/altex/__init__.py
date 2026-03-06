"""Altex - A simple wrapper on top of Altair to make charts with an express API.

If you're lazy and/or familiar with Altair, this is probably a good fit!
Inspired by plost and plotly-express.
"""

__version__ = "0.2.0"
__author__ = "Arnaud Miribel"

# Import chart functions
from .charts import (
    area_chart,
    bar_chart,
    chart,
    hist_chart,
    line_chart,
    scatter_chart,
    sparkarea_chart,
    sparkbar_chart,
    sparkhist_chart,
    sparkline_chart,
)

# Import data utilities
from .data import (
    get_barley_data,
    get_random_data,
    get_stocks_data,
    get_weather_data,
)

__all__ = [
    # Chart functions
    "chart",
    "line_chart",
    "bar_chart",
    "area_chart",
    "scatter_chart",
    "hist_chart",
    "sparkline_chart",
    "sparkbar_chart",
    "sparkhist_chart",
    "sparkarea_chart",
    # Data utilities
    "get_weather_data",
    "get_stocks_data",
    "get_barley_data",
    "get_random_data",
]
