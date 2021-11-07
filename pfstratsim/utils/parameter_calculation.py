import numpy as np
import pandas as pd

from pypfopt import expected_returns as er
from pypfopt import risk_models as rm

DAY_TO_YEAR = 252


def calc_asset_returns(prices):
    """Calculate the returns of the assets.

    Parameters
    ----------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.

    Returns
    -------
    returns : DataFrame of shape (num_times, num_assets) and float
        The historical returns of the assets.
    """
    return er.returns_from_prices(prices)


def calc_corr_cf(prices):
    """Calculate the correlation coefficients of the assets.

    Parameters
    ----------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.

    Returns
    -------
    corr_cf : DataFrame of shape (num_assets, num_assets) and float
        The correlation coefficients of the assets.
    """
    return prices.corr()


def calc_asset_expctd_returns(prices, method="exp", compounding=True, frequency=DAY_TO_YEAR, span=2*DAY_TO_YEAR, dtype="DataFrame", index=None):
    """Calculate the expected returns of the assets.

    Parameters
    ----------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.

    method : {"mean", "exp"}, default "exp"
        The method to calculate the expected returns. If "mean", they are calculated as non-weighted mean; "exp", done
        as exponentially weighted mean.

    compounding : bool, default True
        The method to calculate the mean. If "True", it is geometric mean; else, it is arithmetic mean.

    frequency : int, default DAY_TO_YEAR=252
        The number of days to convert daily rate to an arbitrary rate.

    span : int, default 2*DAY_TO_YEAR
        The parameter to specify the weights used in the calculation of the exponentially weighted returns.

    dtype : {"Series", "DataFrame"}, default "DataFrame"
        The data type of the expected returns of the assets.

    index : list of shape (num_times) and Timestamp, default None
        The index of the expected returns of the assets.

    Returns
    -------
    returns : DataFrame of shape (num_times=1, num_assets) and float
        The expected returns of the assets.
    """
    if method == "exp":
        returns = er.ema_historical_return(prices, compounding=compounding, frequency=frequency, span=span)
    elif method == "mean":
        returns = er.mean_historical_return(prices, compounding=compounding, frequency=frequency)
    else:
        message = f"Invalid value for 'method': {method}." \
                  f"'method' must be in ['exp', 'mean']."
        raise ValueError(message)
    returns = _convert_dtype(returns, dtype, index, prices.columns)
    return returns


def calc_asset_expctd_risks(prices, method="exp", frequency=DAY_TO_YEAR, span=2*DAY_TO_YEAR, dtype="DataFrame", index=None):
    """Calculate the expected risks of the assets.

    Parameters
    ----------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.

    method : {"mean", "exp"}, default "exp"
        The method to calculate the expected risks. If "mean", they are calculated as non-weighted covariances; "exp",
        done as exponentially weighted covariances.

    frequency : int, default DAY_TO_YEAR=252
        The number of days to convert daily rate to an arbitrary time rate.

    span : int, default 2*DAY_TO_YEAR
        The parameter to specify the weights used in the calculation of the exponentially weighted covariances.

    dtype : {"Series", "DataFrame"}, default "DataFrame"
        The data type of the expected risks of the assets.

    index : list of shape (num_times) and Timestamp, default None
        The index of the expected risks of the assets.

    Returns
    -------
    risks : DataFrame of shape (num_times=1, num_assets) and float
        The expected risks of the assets.
    """
    if method == "exp":
        covs = rm.exp_cov(prices, frequency=frequency, span=span)
    elif method == "mean":
        covs = rm.sample_cov(prices, frequency=frequency)
        # covs = prices.pct_change().cov() * frequency
    else:
        message = f"Invalid value for 'method': {method}." \
                  f"'method' must be in ['exp', 'mean']."
        raise ValueError(message)
    risks = np.sqrt(np.diag(covs))
    risks = _convert_dtype(risks, dtype, index, prices.columns)
    return risks


def calc_asset_obsrvd_returns(prices, frequency=DAY_TO_YEAR, dtype="DataFrame", index=None):
    """Calculate the observed returns of the assets.

    Parameters
    ----------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.

    frequency : int, default DAY_TO_YEAR=252
        The number of days to convert daily rate to an arbitrary rate.

    dtype : {"Series", "DataFrame"}, default "DataFrame"
        The data type of the observed returns of the assets.

    index : list of shape (num_times) and Timestamp, default None
        The index of the observed returns of the assets.

    Returns
    -------
    returns : DataFrame of shape (num_times=1, num_assets) and float
        The observed returns of the assets.
    """
    returns = er.mean_historical_return(prices, compounding=False, frequency=frequency)
    returns = _convert_dtype(returns, dtype, index, prices.columns)
    return returns


def calc_asset_obsrvd_risks(prices, frequency=DAY_TO_YEAR, dtype="DataFrame", index=None):
    """Calculate the observed risks of the assets.

    Parameters
    ----------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.

    frequency : int, default DAY_TO_YEAR=252
        The number of days to convert daily rate to an arbitrary rate.

    dtype : {"Series", "DataFrame"}, default "DataFrame"
        The data type of the expected risks of the assets.

    index : list of shape (num_times) and Timestamp, default None
        The index of the observed risks of the assets.

    Returns
    -------
    risks : DataFrame of shape (num_times=1, num_assets) and float
        The observed risks of the assets.
    """
    covs = rm.sample_cov(prices, frequency=frequency)
    risks = np.sqrt(np.diag(covs))
    risks = _convert_dtype(risks, dtype, index, prices.columns)
    return risks


def calc_prtfl_obsrvd_return(asset_props, index=None, columns=None, **kwargs):
    """Calculate the observed return of the portfolio.

    Parameters
    ----------
    asset_props : DataFrame of shape (num_times=1, num_prtfls=1) and float
        The asset proportions calculated by the solver algorithm.

    index : list of shape (num_times) and Timestamp, default None
        The index of the observed return of the portfolio.

    columns : list of shape (num_prtfls=1) and str, default None
        The column of the observed return of the portfolio.

    kwargs : dict
        The parameters to calculate the observed returns of the assets.

    Returns
    -------
    return : DataFrame of shape (num_times=1, num_prtfls=1) and float
        The observed return of the portfolio
    """
    asset_returns = calc_asset_obsrvd_returns(dtype="Series", **kwargs)
    asset_props = asset_props.iloc[0,:]
    num_assets = len(asset_props)
    prtfl_return = 0.0
    for a in range(num_assets):
        prtfl_return += asset_returns[a] * asset_props[a]
    return pd.DataFrame([prtfl_return], index=index, columns=columns)


def calc_prtfl_obsrvd_risk(asset_props, index=None, columns=None, **kwargs):
    """Calculate the observed risk of the portfolio.

    Parameters
    ----------
    asset_props : DataFrame of shape (num_times, num_prtfls=1) and float, default None
        The asset proportions calculated by the solver algorithm.

    index : list of shape (num_times) and Timestamp, default None
        The index of the observed risk of the portfolio.

    columns : list of shape (num_prtfls=1) and str, default None
        The column of the observed risk of the portfolio.

    kwargs : dict
        The parameters to calculate the observed risks and the correlation coefficients of the assets.

    Returns
    -------
    return : DataFrame of shape (num_times=1, num_prtfls=1) and float
        The observed risk of the portfolio
    """
    asset_risks = calc_asset_obsrvd_risks(dtype="Series", **kwargs)
    corr_cf = calc_corr_cf(kwargs.get("prices"))

    asset_props = asset_props.iloc[0, :]
    num_assets = len(asset_props)
    prtfl_var = 0.0
    for a in range(num_assets):
        for a1 in range(num_assets):
            prtfl_var += corr_cf.iloc[a, a1] * asset_risks[a] * asset_risks[a1] * asset_props[a] * asset_props[a1]
    prtfl_risk = np.sqrt(prtfl_var)
    return pd.DataFrame([prtfl_risk], index=index, columns=columns)


def _convert_dtype(data, dtype=None, index=None, columns=None):
    """Convert "Series" type to "DataFrame" and vice versa.

    Parameters
    ----------
    data : DataFrame of shape (num_times, num_assets=1) and float, default None
        The data to convert to a different type.

    dtype : {"Series", "DataFrame"}, default None
        The data type of the converted data.

    index : list of shape (num_times) and Timestamp, default None
        The index of the converted data.

    columns : list of shape (num_assets) and str, default None
        The columns of the converted data.

    Returns
    -------
    data : Series of shape (num_assets) or DataFrame of shape (num_times=1, num_assets) and float
        The converted data.
    """
    if dtype == "Series":
        data = pd.Series(data, index=columns)
    elif dtype == "DataFrame":
        data = pd.DataFrame(data, index=columns).T
        if index is not None:
            data.index = index
    else:
        message = f"Invalid value for 'dtype': {dtype}." \
                  f"'dtype' must be in ['Series', 'DataFrame]."
        raise ValueError(message)
    return data
