"""
plotting_utils.py
-----------------
Utility functions for time series plotting and diagnostics using Plotly.

This module provides functions for visualizing time series data, including:
    - ACF/PACF plots
    - Residual diagnostics
    - Series and forecast visualization
    - Decomposition and ETS component plots

Dependencies: numpy, pandas, polars, plotly, statsmodels, utilsforecast
"""

import polars as pl
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objects as go
from utilsforecast.plotting import plot_series, DFType
from typing import Optional, Union, Any


def _plot(
    values: np.ndarray,
    confidence_interval: np.ndarray,
    title: str,
    fig: Optional[go.Figure] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Helper function to plot values with confidence intervals as vertical lines and shaded area.
    """
    if fig is None:
        fig = make_subplots()

    for x in range(len(values)):
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[0, values[x]],
                mode="lines",
                line=dict(color="#3f3f3f", width=1),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.add_scatter(
        x=np.arange(len(values)),
        y=values,
        mode="markers",
        marker_color="black",
        marker_size=6,
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(values)),
            y=confidence_interval[:, 0] - values,
            mode="lines",
            line=dict(color="rgba(255,255,255,0)", width=1),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(values)),
            y=confidence_interval[:, 1] - values,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(169, 169, 169, 0.3)",  # Light grey color
            line=dict(color="rgba(255,255,255,0)", width=1),
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    fig.update_layout(
        title=title,
        showlegend=False,
    )

    return fig


def plot_acf(
    data: Any,
    adjusted: bool = False,
    nlags: Optional[int] = None,
    qstat: bool = False,
    fft: bool = True,
    alpha: float = 0.05,
    bartlett_confint: bool = True,
    missing: str = "none",
    title: str = "Autocorrelation Function (ACF)",
    fig: Optional[go.Figure] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Plot the autocorrelation function (ACF) for a time series.
    """
    acf_values, confidence_interval = acf(
        data,
        adjusted=adjusted,
        nlags=nlags,
        qstat=qstat,
        fft=fft,
        alpha=alpha,
        bartlett_confint=bartlett_confint,
        missing=missing,
    )

    fig = _plot(
        acf_values,
        confidence_interval,
        title=title,
        fig=fig,
        row=row,
        col=col,
    )

    return fig


def plot_pacf(
    data: Any,
    nlags: Optional[int] = None,
    alpha: float = 0.05,
    method: str = "yw",
    title: str = "Partial Autocorrelation Function (PACF)",
    fig: Optional[go.Figure] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> go.Figure:
    """
    Plot the partial autocorrelation function (PACF) for a time series.
    """
    pacf_values, confidence_interval = pacf(
        data,
        nlags=nlags,
        alpha=alpha,
        method=method,
    )

    _plot(
        pacf_values,
        confidence_interval,
        title=title,
        fig=fig,
        row=row,
        col=col,
    )

    return fig


def plot_acf_pacf(
    data: Any,
    adjusted: bool = False,
    nlags: Optional[int] = None,
    qstat: bool = False,
    fft: bool = True,
    alpha: float = 0.05,
    bartlett_confint: bool = True,
    missing: str = "none",
    pacf_method: str = "yw",
    title: Optional[str] = None,
    title_acf: str = "Autocorrelation Function (ACF)",
    title_pacf: str = "Partial Autocorrelation Function (PACF)",
    fig: Optional[go.Figure] = None,
    row: Optional[int] = None,
) -> go.Figure:
    """
    Plot both ACF and PACF for a time series in a single figure.
    """
    if fig is None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(title_acf, title_pacf),
            horizontal_spacing=0.1,
        )
        row = 1
        fig.update_layout(
            template="plotly_white",
        )

    row = row or 1

    fig = plot_acf(
        data,
        adjusted=adjusted,
        nlags=nlags,
        qstat=qstat,
        fft=fft,
        alpha=alpha,
        bartlett_confint=bartlett_confint,
        missing=missing,
        title=None,
        fig=fig,
        row=row,
        col=1,
    )

    fig = plot_pacf(
        data,
        nlags=nlags,
        alpha=alpha,
        method=pacf_method,
        title=None,
        fig=fig,
        row=row,
        col=2,
    )
    fig.update_layout(title=title)

    return fig


def plot_series_acf_pacf(
    data: Any,
    time: Optional[Any] = None,
    adjusted: bool = False,
    nlags: Optional[int] = None,
    qstat: bool = False,
    fft: bool = True,
    alpha: float = 0.05,
    bartlett_confint: bool = True,
    missing: str = "none",
    pacf_method: str = "yw",
    title: Optional[str] = None,
    title_acf: str = "Autocorrelation Function (ACF)",
    title_pacf: str = "Partial Autocorrelation Function (PACF)",
    width: int = 1200,
    height: int = 600,
) -> go.Figure:
    """
    Plot the time series, ACF, and PACF in a 2x2 subplot layout.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(title, title_acf, title_pacf),
        specs=[[{"colspan": 2}, None], [{}, {}]],
        horizontal_spacing=0.1,
    )
    if isinstance(data, (pd.Series, pl.Series)):
        x = np.arange(len(data))
        y = data
    else:
        x = data["ds"]
        y = data["y"]

    if time is not None:
        x = time

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            line=dict(color="black", width=1),
        ),
        row=1,
        col=1,
    )

    plot_acf_pacf(
        data=y,
        adjusted=adjusted,
        nlags=nlags,
        qstat=qstat,
        fft=fft,
        alpha=alpha,
        bartlett_confint=bartlett_confint,
        missing=missing,
        pacf_method=pacf_method,
        fig=fig,
        row=2,
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=10),
        width=width,
        height=height,
    )

    return fig


def plot_residuals_diagnostic(
    residuals: np.ndarray,
    time: Any,
    adjusted: bool = False,
    nlags: Optional[int] = None,
    qstat: bool = False,
    fft: bool = True,
    alpha: float = 0.05,
    bartlett_confint: bool = True,
    missing: str = "none",
    width: int = 1200,
    height: int = 600,
) -> go.Figure:
    """
    Plot residual diagnostics: residuals, ACF, and histogram.
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Innovation Residuals", "ACF Plot", "Histogram"),
        specs=[[{"colspan": 2}, None], [{}, {}]],
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=residuals,
            mode="lines",
            name="Residuals",
            line=dict(color="black", width=1),
        ),
        row=1,
        col=1,
    )

    plot_acf(
        data=residuals,
        adjusted=adjusted,
        nlags=nlags,
        qstat=qstat,
        fft=fft,
        alpha=alpha,
        bartlett_confint=bartlett_confint,
        missing=missing,
        title=None,
        fig=fig,
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            name="Residuals",
            marker=dict(color="black", line=dict(color="black")),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(size=10),
        width=width,
        height=height,
    )

    return fig


def plotly_series(
    df: Optional[DFType] = None,
    forecasts_df: Optional[DFType] = None,
    ids: Optional[list[str]] = None,
    plot_random: bool = True,
    max_ids: int = 8,
    models: Optional[list[str]] = None,
    level: Optional[list[float]] = None,
    max_insample_length: Optional[int] = None,
    plot_anomalies: bool = False,
    palette: Optional[str] = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    seed: int = 0,
    resampler_kwargs: Optional[dict] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[Union[list[str], str]] = None,
    width: int = 1200,
    height: int = 500,
    legend: bool = True,
    date_range: Optional[tuple[str, str]] = None,
) -> go.Figure:
    """
    Plot time series and forecasts using Plotly, with support for multiple series and models.
    """
    fig = plot_series(
        df=df,
        forecasts_df=forecasts_df,
        ids=ids,
        plot_random=plot_random,
        max_ids=max_ids,
        models=models,
        level=level,
        max_insample_length=max_insample_length,
        plot_anomalies=plot_anomalies,
        palette=palette,
        id_col=id_col,
        time_col=time_col,
        target_col=target_col,
        seed=seed,
        resampler_kwargs=resampler_kwargs,
        engine="plotly",
    )

    n_subplots = len(fig.layout.annotations) - 2
    if xlabel:
        fig.layout.annotations[-2].update(text=xlabel)
    if ylabel:
        fig.layout.annotations[-1].update(text=ylabel)

    if title is not None:
        if isinstance(title, str):
            title = [title] * n_subplots

        assert n_subplots == len(title), "Length of title must match number of series"

        for i, t in enumerate(title):
            fig.layout.annotations[i].update(text=t)

    if date_range:
        fig.update_xaxes(type="date", range=date_range)

    fig.update_layout(showlegend=legend, width=width, height=height)

    return fig


def plot_real_data_vs_insample_forecast(
    y: Any,
    y_hat: Any,
    title: Optional[str] = None,
    width: int = 1200,
    height: int = 400,
) -> go.Figure:
    """
    Plot actual vs fitted values for in-sample forecast diagnostics.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y,
            y=y_hat,
            mode="markers",
            marker=dict(color="black"),
            name="Fitted vs Actual",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min(y), max(y)],
            y=[min(y_hat), max(y_hat)],
            mode="lines",
            line=dict(dash="dash", color="blue"),
            name="y = x",
        )
    )
    fig.update_layout(
        xaxis_title="Data (actual values)",
        yaxis_title="Fitted (predicted values)",
        title=title,
        template="plotly_white",
        width=width,
        height=height,
    )

    return fig


def plot_decomposition(
    time: Any,
    observed: Optional[np.ndarray] = None,
    seasonal: Optional[Union[np.ndarray, dict]] = None,
    trend: Optional[np.ndarray] = None,
    resid: Optional[np.ndarray] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """
    Plots the decomposition output.
    Supports multiple seasonalities if `seasonal` is a dict of {name: array} or multidim array.
    """
    """
    Plots the decomposition output.
    Supports multiple seasonalities if `seasonal` is a dict of {name: array} or multidim array.
    """

    series = []
    if observed is not None:
        series.append("Original")
    if trend is not None:
        series.append("Trend")
    # Handle multiple seasonalities
    seasonal_names = []
    if isinstance(seasonal, np.ndarray) and seasonal.ndim > 1:
        seasonal = {f"{i + 1}": seasonal[:, i] for i in range(seasonal.shape[1])}
    if isinstance(seasonal, dict):
        for k in seasonal:
            series.append(f"Seasonal ({k})")
            seasonal_names.append(k)
    elif seasonal is not None:
        series.append("Seasonal")
        seasonal_names.append(None)
    if resid is not None:
        series.append("Residual")

    if len(series) == 0:
        raise ValueError(
            "All component flags were off. Need at least one of the flags turned on to plot."
        )

    fig = make_subplots(
        rows=len(series), cols=1, shared_xaxes=True, subplot_titles=series
    )
    x = time
    row = 1
    if observed is not None:
        fig.add_trace(go.Scatter(x=x, y=observed, name="Original"), row=row, col=1)
        row += 1
    if trend is not None:
        fig.add_trace(go.Scatter(x=x, y=trend, name="Trend"), row=row, col=1)
        row += 1
    if isinstance(seasonal, dict):
        for k in seasonal_names:
            fig.add_trace(
                go.Scatter(x=x, y=seasonal[k], name=f"Seasonal {k}"),
                row=row,
                col=1,
            )
            row += 1
    elif seasonal is not None:
        fig.add_trace(
            go.Scatter(x=x, y=seasonal, name="Seasonal"),
            row=row,
            col=1,
        )
        row += 1
    if resid is not None:
        fig.add_trace(go.Scatter(x=x, y=resid, name="Residual"), row=row, col=1)
        row += 1

    fig.update_layout(
        title_text="Seasonal Trend Decomposition",
        autosize=False,
        width=width or 1200,
        height=height or (200 * len(series) + 100),
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        legend_title=None,
        showlegend=False,
        legend=dict(
            font=dict(size=15),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
    )
    return fig


def plot_ets_components(
    time: Union[np.ndarray, list],
    observed: Optional[np.ndarray] = None,
    level: Optional[np.ndarray] = None,
    slope: Optional[np.ndarray] = None,
    season: Optional[np.ndarray] = None,
    resid: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """
    Plots ETS decomposition components: observed, level, slope, season, residual.
    """
    """
    Plots ETS decomposition components: observed, level, slope, season, residual.
    """
    series = []
    if observed is not None:
        series.append("Observed")
    if level is not None:
        series.append("Level")
    if slope is not None:
        series.append("Slope")
    if season is not None:
        series.append("Season")
    if resid is not None:
        series.append("Residual")

    if len(series) == 0:
        raise ValueError("At least one component must be provided to plot.")

    fig = make_subplots(
        rows=len(series), cols=1, shared_xaxes=True, subplot_titles=series
    )
    x = time
    row = 1
    if observed is not None:
        fig.add_trace(go.Scatter(x=x, y=observed, name="Observed"), row=row, col=1)
        row += 1
    if level is not None:
        fig.add_trace(go.Scatter(x=x, y=level, name="Level"), row=row, col=1)
        row += 1
    if slope is not None:
        fig.add_trace(go.Scatter(x=x, y=slope, name="Slope"), row=row, col=1)
        row += 1
    if season is not None:
        fig.add_trace(go.Scatter(x=x, y=season, name="Season"), row=row, col=1)
        row += 1
    if resid is not None:
        fig.add_trace(go.Scatter(x=x, y=resid, name="Residual"), row=row, col=1)
        row += 1

    fig.update_layout(
        title_text=title or "ETS Components",
        autosize=False,
        width=width or 1200,
        height=height or (200 * len(series) + 100),
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        showlegend=False,
    )
    return fig
