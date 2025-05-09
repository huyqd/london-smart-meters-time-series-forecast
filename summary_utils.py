import polars as pl
import pandas as pd
import numpy as np
from scipy import stats
from statsforecast.arima import ARIMASummary, Arima
from statsforecast import StatsForecast
from mlforecast import MLForecast
from copy import deepcopy


def print_arima_fitted_summary(fitted_model: Arima):
    """
    Print the summary of the fitted ARIMA model.

    Parameters
    ----------
    fitted_model : Arima
        The fitted ARIMA model object.
    """
    print(ARIMASummary(fitted_model))
    coefs = deepcopy(fitted_model["coef"])
    print("Coefficients:")
    for param, value in coefs.items():
        print(f" {param}{(11 - len(param)) * ' '}: {value:.2f}")
    print(f"sigma^2     : {fitted_model['sigma2']:.2f}")
    print(f"loglik      : {fitted_model['loglik']:.2f}")
    print(f"aic         : {fitted_model['aic']:.2f}")
    print(f"aicc        : {fitted_model['aicc']:.2f}")
    print(f"bic         : {fitted_model['bic']:.2f}")


def arima_fitted_summary_dataframe(fcst: StatsForecast):
    summaries = []
    for model in fcst.fitted_[0]:
        summary_model = {
            "model": model,
            "Orders": ARIMASummary(model.model_),
            "sigma2": model.model_["sigma2"],
            "loglik": model.model_["loglik"],
            "aic": model.model_["aic"],
            "aicc": model.model_["aicc"],
            "bic": model.model_["bic"],
        }

        summaries.append(summary_model)

    return pl.DataFrame(sorted(summaries, key=lambda d: d["aicc"]))


def print_regression_summary_from_model(
    model,
    X: pl.DataFrame | pd.DataFrame,
    y: pl.Series | pd.Series,
):
    if isinstance(X, pl.DataFrame):
        X = X.to_pandas()
    if isinstance(y, pl.Series):
        y = y.to_pandas()

    feature_names = X.columns.tolist()
    residuals = y - model.predict(X)

    num_obs, num_features = X.shape
    X_design = np.hstack([np.ones((num_obs, 1)), X]).astype(np.float64)
    degrees_of_freedom = num_obs - num_features - 1

    res_summary = np.percentile(residuals, [0, 25, 50, 75, 100])
    print("#> Residuals:")
    print("#>     Min      1Q  Median      3Q     Max ")
    print(
        f"#> {res_summary[0]:7.4f} {res_summary[1]:7.4f} {res_summary[2]:7.4f} "
        f"{res_summary[3]:7.4f} {res_summary[4]:7.4f}\n"
    )

    coef = np.insert(model.coef_, 0, model.intercept_)
    rss = np.sum(residuals**2)
    mse = rss / degrees_of_freedom
    var_beta = mse * np.linalg.inv(X_design.T @ X_design).diagonal()
    se_beta = np.sqrt(var_beta)
    t_stats = coef / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), degrees_of_freedom))

    print("#> Coefficients:")
    print(
        f"#> {'':>13} {'Estimate':>9} {'Std. Error':>11} {'t value':>9} {'Pr(>|t|)':>9}"
    )
    names = ["(Intercept)"] + feature_names
    for name, est, se, t, p in zip(names, coef, se_beta, t_stats, p_values):
        stars = (
            "***"
            if p < 0.001
            else "**"
            if p < 0.01
            else "*"
            if p < 0.05
            else "."
            if p < 0.1
            else ""
        )
        print(f"#> {name:>13} {est:9.4f} {se:11.4f} {t:9.2f} {p:9.3g} {stars}")
    print("---")
    print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    r_squared = model.score(X, y)
    adj_r_squared = 1 - (1 - r_squared) * (num_obs - 1) / degrees_of_freedom
    f_stat = (r_squared / num_features) / ((1 - r_squared) / degrees_of_freedom)
    f_pval = 1 - stats.f.cdf(f_stat, num_features, degrees_of_freedom)

    print(
        f"\nResidual standard error: {np.sqrt(mse):.3f} on {degrees_of_freedom} degrees of freedom"
    )
    print(
        f"Multiple R-squared: {r_squared:.3f},   Adjusted R-squared: {adj_r_squared:.3f}"
    )
    print(
        f"F-statistic: {f_stat:.1f} on {num_features} and {degrees_of_freedom} DF, p-value: {f_pval:.3g}"
    )


def get_fitted_residuals(
    fcst: StatsForecast | MLForecast,
    target_col: str = "y",
    time_col: str = "ds",
    id_col: str = "unique_id",
):
    insample_forecasts = fcst.forecast_fitted_values()
    model_names = [
        c for c in insample_forecasts.columns if c not in [target_col, time_col, id_col]
    ]

    if isinstance(insample_forecasts, pd.DataFrame):
        residuals = insample_forecasts.copy(deep=True)
        residuals[model_names] = residuals[model_names].sub(
            insample_forecasts[target_col], axis=0
        )[[time_col, id_col] + model_names]
    elif isinstance(insample_forecasts, pl.DataFrame):
        residuals = insample_forecasts.select(
            time_col, id_col, pl.col(model_names) - pl.col(target_col)
        )
    else:
        raise ValueError("Unsupported DataFrame type")

    return residuals
