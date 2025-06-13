from utilsforecast.losses import _pl_agg_expr
from utilsforecast.compat import DFType
import polars as pl
import pandas as pd


def winkler_score(
    df: DFType,
    models,
    level: int,
    id_col: str = "unique_id",
    target_col: str = "y",
) -> DFType:
    """
    Compute the Winkler score for probabilistic forecasts.
    https://otexts.com/fpp3/distaccuracy.html#winkler-score

    Args:
        df: Input data as a Polars DataFrame.
        models: List of model names (str) to evaluate.
        level (int): Prediction interval level (e.g., 80 for 80% interval).
        id_col (str): Name of the column identifying unique series (default: "unique_id").
        target_col (str): Name of the target column (default: "y").

    Returns:
        Polars DataFrame with Winkler scores for each model and id.
    """
    if isinstance(df, pd.DataFrame):
        raise NotImplementedError
    else:

        def gen_expr(model):
            upper_level = pl.col(f"{model}-hi-{level}")
            lower_level = pl.col(f"{model}-lo-{level}")
            target = pl.col(target_col)
            sharpness = upper_level - lower_level
            alpha = 1 - (level / 100)
            lower_calibration = (
                2
                / alpha
                * (lower_level - target)
                * (target < lower_level).cast(pl.Float32)
            )
            upper_calibration = (
                2
                / alpha
                * (target - upper_level)
                * (target > upper_level).cast(pl.Float32)
            )
            return (sharpness + lower_calibration + upper_calibration).alias(model)

        res = _pl_agg_expr(df, models, id_col, gen_expr)

    return res
