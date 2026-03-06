"""Chart functions for Altex - a simple wrapper on top of Altair."""

from functools import partial
from typing import Optional, Union

import altair as alt
import pandas as pd
import streamlit as st

# Handle Altair theme configuration
try:
    from altair.utils.plugin_registry import NoSuchEntryPoint
except ImportError:
    from entrypoints import NoSuchEntryPoint

try:
    alt.themes.enable("streamlit")
except NoSuchEntryPoint:
    st.altair_chart = partial(st.altair_chart, theme="streamlit")


def _drop_nones(iterable: Union[dict, list]):
    """Remove nones for iterable.
    If dict, drop keys when value is None
    If list, drop values when value equal None

    Args:
        iterable: Input iterable

    Raises:
        TypeError: When iterable type is not supported

    Returns:
        Input iterable without Nones
    """
    if isinstance(iterable, dict):
        return {k: v for k, v in iterable.items() if v is not None}
    if isinstance(iterable, list):
        return [x for x in iterable if x is not None]
    raise TypeError(f"Iterable of type {type(iterable)} is not supported")


def _get_shorthand(param: Union[str, alt.X, alt.Y]) -> Optional[str]:
    """Get Altair shorthand from parameter, if exists.

    Args:
        param: Parameter x/y

    Returns:
        Parameter itself or shorthand when alt.X/alt.Y object
    """
    if param is None:
        return None
    if not isinstance(param, str):
        return param.shorthand
    return param


def _update_axis_config(
    axis: Union[alt.X, alt.Y, str], output_type: Union[alt.X, alt.Y], updates: dict
) -> Union[alt.X, alt.Y]:
    """Update x and y configs.

    Args:
        axis: Chart input for x
        output_type: Chart input for y
        updates: Dictionary of updates to apply

    Raises:
        TypeError: When input has invalid type

    Returns:
        Updated config for x/y
    """
    if isinstance(axis, (alt.X, alt.Y)):
        axis_config = axis.to_dict()
        for key, value in updates.items():
            axis_config[key] = value
        return output_type(**axis_config)
    if isinstance(axis, str):
        return output_type(shorthand=axis, **updates)
    raise TypeError("Input x/y must be of type str or alt.X or alt.Y")


def _chart(
    mark_function: str,
    data: pd.DataFrame,
    x: Union[alt.X, str],
    y: Union[alt.Y, str],
    color: Optional[Union[alt.Color, str]] = None,
    opacity: Optional[Union[alt.value, float]] = None,
    column: Optional[Union[alt.Column, str]] = None,
    rolling: Optional[int] = None,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    spark: bool = False,
    autoscale_y: bool = False,
) -> alt.Chart:
    """Create an Altair chart with a simple API.

    Supported charts include line, bar, point, area, histogram, sparkline,
    sparkbar, sparkarea.

    Args:
        mark_function: Altair mark function, example line/bar/point
        data: Dataframe to use for the chart
        x: Column for the x axis
        y: Column for the y axis
        color: Color a specific group of your data. Defaults to None.
        opacity: Change opacity of marks. Defaults to None.
        column: Groupby a specific column. Defaults to None.
        rolling: Rolling average window size. Defaults to None.
        title: Title of the chart. Defaults to None.
        width: Width of the chart. Defaults to None.
        height: Height of the chart. Defaults to None.
        spark: Whether or not to make spark chart, i.e. a chart without
            axes nor ticks nor legend. Defaults to False.
        autoscale_y: Whether or not to autoscale the y axis. Defaults to False.

    Returns:
        Altair chart
    """
    x_ = _get_shorthand(x)
    y_ = _get_shorthand(y)
    color_ = _get_shorthand(color)

    tooltip_config = _drop_nones([x_, y_, color_])

    chart_config = _drop_nones(
        {
            "data": data,
            "title": title,
            "mark": mark_function,
            "width": width,
            "height": height,
        }
    )

    chart = alt.Chart(**chart_config)

    if rolling is not None:
        rolling_column = f"{y_} ({rolling}-average)"
        y = f"{rolling_column}:Q"
        transform_config = {
            rolling_column: f"mean({y_})",
            "frame": [-rolling, 0],
            "groupby": [str(color)],
        }
        chart = chart.transform_window(**transform_config)

    if spark:
        chart = chart.configure_view(strokeWidth=0).configure_axis(
            grid=False, domain=False
        )
        x_axis = _update_axis_config(x, alt.X, {"axis": None})
        y_axis = _update_axis_config(y, alt.Y, {"axis": None})
    else:
        x_axis = x
        y_axis = y

    if autoscale_y:
        y_axis = _update_axis_config(y_axis, alt.Y, {"scale": alt.Scale(zero=False)})

    encode_config = _drop_nones(
        {
            "x": x_axis,
            "y": y_axis,
            "color": color,
            "tooltip": tooltip_config,
            "opacity": alt.value(opacity) if isinstance(opacity, float) else opacity,
            "column": column,
        }
    )

    return chart.encode(**encode_config)


def chart(use_container_width: bool = True, **kwargs) -> None:
    """Display an Altair chart in Streamlit.

    Args:
        use_container_width: Whether or not the displayed chart uses all
            available width. Defaults to True.
        **kwargs: See function _chart()
    """
    chart_obj = _chart(**kwargs)

    if "width" in kwargs:
        use_container_width = False

    st.altair_chart(
        chart_obj,
        use_container_width=use_container_width,
    )


def _partial(*args, **kwargs):
    """Alternative to 'functools.partial' where __name__ attribute
    can be set manually, since the default partial does not create it.
    """
    __name__ = kwargs.pop("__name__", "foo")
    func = partial(*args, **kwargs)
    func.__name__ = __name__
    return func


def scatter_chart(**kwargs) -> None:
    """Create a scatter chart.

    Args:
        **kwargs: See function _chart()
    """
    return chart(mark_function="point", **kwargs)


# Create chart functions using partial
line_chart = _partial(chart, mark_function="line", __name__="line_chart")
area_chart = _partial(chart, mark_function="area", __name__="area_chart")
bar_chart = _partial(chart, mark_function="bar", __name__="bar_chart")
hist_chart = _partial(bar_chart, y="count()", __name__="hist_chart")
sparkline_chart = _partial(line_chart, spark=True, __name__="sparkline_chart")
sparkbar_chart = _partial(bar_chart, spark=True, __name__="sparkbar_chart")
sparkhist_chart = _partial(hist_chart, spark=True, __name__="sparkhist_chart")
sparkarea_chart = _partial(area_chart, spark=True, __name__="sparkarea_chart")
