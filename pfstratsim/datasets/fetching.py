import os

import pandas as pd
from pandas_datareader import data


def load_sample_prices(asset_name_list, start_time, end_time, interest_rate=None, **params):
    """Load the historical prices of the sample assets.

    Parameters
    ----------
    asset_name_list : list of shape (num_assets) and str
        The name list of the assets.

    start_time=None : Timestamp
        The start time for the simulation.

    end_time=None : Timestamp
        The end time for the simulation.

    interest_rate=None : float, default None
        The interest rate of cash.

    params : dict
        The parameters not to be used in this method but necessary just to realize the API that can call this method by
        one way.

    Returns
    -------
    sample_prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.
    """
    data_dir = os.path.join(os.path.dirname(os.path.normpath(__file__)), "data")
    sample_prices = pd.DataFrame()
    for asset_name in asset_name_list:
        all_data = pd.read_csv(os.path.join(data_dir, f"{asset_name}.csv"), index_col="Date", parse_dates=["Date"])
        sample_prices[asset_name] = all_data["Close"][start_time:end_time]

    if interest_rate is not None:
        sample_prices = _add_cash_data(sample_prices, interest_rate)

    return sample_prices


def fetch_prices(asset_name_list, start_time, end_time, interest_rate=None, kind="Close", save_dir=".", **params):
    """Load the historical prices of the arbitrary assets.

    Parameters
    ----------
    asset_name_list : list of shape (num_assets) and str
        The name list of the assets.

    start_time : Timestamp
        The start time for the simulation.

    end_time : Timestamp
        The end time for the simulation.

    interest_rate : float, default None
        The interest rate of cash.

    kind : {"Open", "High", "Low", "Close"}, default "Close"
        The kind of the prices.

    save_dir : str, default "."
        The directory of the historical prices of the assets.

    params : dict
        The parameters not to be used in this method but necessary just to realize the API that can call this method by
        one way.

    Returns
    -------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.
    """
    prices = pd.DataFrame()
    for asset_name in asset_name_list:
        all_data = data.DataReader(asset_name, "yahoo", start_time, end_time)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            all_data.to_csv(os.path.join(save_dir, f"{asset_name}.csv"))
        prices[asset_name] = all_data[kind]

    if interest_rate is not None:
        prices = _add_cash_data(prices, interest_rate)

    return prices


def _add_cash_data(prices, interest_rate):
    """Add the historical prices of cash to the ones of the assets.

    Parameters
    ----------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.

    interest_rate : float
        The interest rate of cash.

    Returns
    -------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets with ones of cash.
    """
    cash_list = [1.0]  # init cash price doesn't affect its return and risk
    for i in range(len(prices) - 1):
        cash_list.append(cash_list[-1] * (1.0 + interest_rate / len(prices)))
    prices['CASH'] = cash_list
    return prices
