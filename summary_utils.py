import polars as pl
import pandas as pd
import numpy as np
from statsforecast.arima import ARIMASummary
from statsforecast import StatsForecast
from copy import deepcopy


def print_arima_fitted_summary(fitted_model):
    """
    Print the summary of the fitted ARIMA model.

    Parameters
    ----------
    sf : StatsForecast
        The fitted ARIMA model object.
    """
    print(ARIMASummary(fitted_model))
    coefs = deepcopy(fitted_model["coef"])
    coefs = {"itc": coefs.pop("intercept")} | coefs
    print("Coefficients:")
    for param, value in coefs.items():
        print(f" {param}        : {value:.2f}")
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
