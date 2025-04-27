import numpy as np
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
import plotly.graph_objects as go
from utilsforecast.plotting import plot_series, DFType


def _plot(
    values: np.ndarray,
    confidence_interval: np.ndarray,
    title: str,
    fig=None,
    row=None,
    col=None,
):
    if fig is None:
        fig = make_subplots()

    for x in range(len(values)):
        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[0, values[x]],
                mode="lines",
                line=dict(color="#3f3f3f"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    fig.add_scatter(
        x=np.arange(len(values)),
        y=values,
        mode="markers",
        marker_color="#1f77b4",
        marker_size=6,
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(values)),
            y=confidence_interval[:, 0] - values,
            mode="lines",
            line=dict(color="rgba(255,255,255,0)"),
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
            fillcolor="rgba(32, 146, 230,0.3)",
            line=dict(color="rgba(255,255,255,0)"),
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
    data,
    adjusted=False,
    nlags=None,
    qstat=False,
    fft=True,
    alpha=0.05,
    bartlett_confint=True,
    missing="none",
    title=None,
    fig=None,
    row=None,
    col=None,
):
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

    title = title or "Autocorrelation Function (ACF)"

    _plot(
        acf_values,
        confidence_interval,
        title=title,
        fig=fig,
        row=row,
        col=col,
    )

    return fig


def plot_pacf(
    data,
    nlags=None,
    alpha=0.05,
    method="yw",
    title=None,
    fig=None,
    row=None,
    col=None,
):
    pacf_values, confidence_interval = pacf(
        data,
        nlags=nlags,
        alpha=alpha,
        method=method,
    )

    title = title or "Partial Autocorrelation Function (PACF)"

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
    data,
    adjusted=False,
    nlags=None,
    qstat=False,
    fft=True,
    alpha=0.05,
    bartlett_confint=True,
    missing="none",
    pacf_method="yw",
    title_acf=None,
    title_pacf=None,
    fig=None,
):
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=(title_acf, title_pacf), horizontal_spacing=0.1
    )

    fig = plot_acf(
        data,
        adjusted=adjusted,
        nlags=nlags,
        qstat=qstat,
        fft=fft,
        alpha=alpha,
        bartlett_confint=bartlett_confint,
        missing=missing,
        title=title_acf or "Autocorrelation Function (ACF)",
        fig=fig,
        row=1,
        col=1,
    )

    fig = plot_pacf(
        data,
        nlags=nlags,
        alpha=alpha,
        method=pacf_method,
        title=title_pacf or "Partial Autocorrelation Function (PACF)",
        fig=fig,
        row=1,
        col=2,
    )

    return fig


def plotly_series(
    df: DFType | None = None,
    forecasts_df: DFType | None = None,
    ids: list[str] | None = None,
    plot_random: bool = True,
    max_ids: int = 8,
    models: list[str] | None = None,
    level: list[float] | None = None,
    max_insample_length: int | None = None,
    plot_anomalies: bool = False,
    palette: str | None = None,
    id_col: str = "unique_id",
    time_col: str = "ds",
    target_col: str = "y",
    seed: int = 0,
    resampler_kwargs: dict | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: list[str] | str | None = None,
    legend: bool = True,
):
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

    if xlabel:
        fig.layout.annotations[-2].update(text=xlabel)
    if ylabel:
        fig.layout.annotations[-1].update(text=ylabel)

    if isinstance(title, str):
        title = [title] * len(fig.data)

    assert len(fig.data) == len(title), "Length of title must match number of series"

    for i, t in enumerate(title):
        fig.layout.annotations[i].update(text=t)

    fig.update_layout(
        showlegend=legend,
    )

    return fig
