import os
import numpy as np
import pandas as pd
from datetime import timedelta
import joblib
import warnings

from ..problems import RiskMinimization
from ..solvers import Solver, EqualProportion
from ..triggers import Trigger, RegularBasis
from ..utils import calc_asset_obsrvd_returns, calc_asset_obsrvd_risks, calc_prtfl_obsrvd_return, calc_prtfl_obsrvd_risk

warnings.filterwarnings("ignore")


class Simulation(object):
    """The simulation for the portfolio strategy.

    Parameters
    ----------
    prices : DataFrame of shape (num_times, num_assets) and float
        The historical prices of the assets.

    start_time : Timestamp
        The start time for the simulation.

    end_time : Timestamp
        The end time for the simulation.

    init_prtfl_valtn : float, default 100.0
        The initial valuation of the portfolio.

    window_day : int, default 28
        The number of days to specify the population.

    min_reblncng_intrvl_day : int, default 1
        The minimum interval for rebalancing.

    result_dir : str, default "."
        The output directory for the simulation result.

    params : dict
        The other parameters of the trigger, the problem and the solver.
    """
    def __init__(self, prices, start_time, end_time, init_prtfl_valtn=100.0,
                 window_day=28, min_reblncng_intrvl_day=1, result_dir=".", **params):
        self._prices = prices
        self._start_time = start_time
        self._end_time = end_time
        self._init_prtfl_valtn = init_prtfl_valtn
        self._window_day = window_day
        self._min_reblncng_intrvl_day = min_reblncng_intrvl_day
        self._result_dir = result_dir
        self._params = params

    def execute(self):
        """Execute the simulation."""
        # Prepare for storing historical data.
        data_history = {}
        data_history["prices"] = self._prices
        data_history["asset_expctd_returns"] = pd.DataFrame()
        data_history["asset_expctd_risks"] = pd.DataFrame()
        data_history["asset_expctd_valtns"] = pd.DataFrame()
        data_history["prtfl_expctd_return"] = pd.DataFrame()
        data_history["prtfl_expctd_risk"] = pd.DataFrame()
        data_history["prtfl_expctd_valtn"] = pd.DataFrame()

        data_history["asset_obsrvd_returns"] = pd.DataFrame()
        data_history["asset_obsrvd_risks"] = pd.DataFrame()
        data_history["asset_obsrvd_valtns"] = pd.DataFrame()
        data_history["prtfl_obsrvd_return"] = pd.DataFrame()
        data_history["prtfl_obsrvd_risk"] = pd.DataFrame()
        data_history["prtfl_obsrvd_valtn"] = pd.DataFrame()

        data_history["asset_returns"] = pd.DataFrame()
        data_history["asset_valtns"] = pd.DataFrame()
        data_history["asset_valtns_reblncd"] = pd.DataFrame()
        data_history["prtfl_return"] = pd.DataFrame()
        data_history["prtfl_valtn"] = pd.DataFrame()
        data_history["asset_props"] = pd.DataFrame()

        self._params["return_lower_qntl"] = 0.7
        self._params["reblncng_intrvl_day"] = 28
        trigger = Trigger(RegularBasis(**self._params))
        problem = RiskMinimization(**self._params)
        solver = Solver(EqualProportion(**self._params))

        # Set objects for the simulation.
        prev_prices = None
        reblncng_time_list = []
        window = timedelta(days=self._window_day)
        crnt_time = self._start_time + window
        # Execute the simulation.
        while crnt_time <= self._end_time:
            # Ignore date-times that are not included in the prices' date-times.
            if not crnt_time in self._prices.index:
                crnt_time += timedelta(days=1)
                continue
            # Extract the price data in the moving window from the current date-time
            oldest_time = crnt_time - window
            latest_time = crnt_time - timedelta(days=1)  # -1 day due to unknown prices at the current date-time.
            crnt_prices = self._prices[oldest_time:latest_time]

            # Assess the necessity of rebalancing.
            is_reblncng = trigger.assess(crnt_time=crnt_time, reblncng_time_list=reblncng_time_list)
            if not is_reblncng:
                crnt_time += timedelta(days=self._min_reblncng_intrvl_day)
                continue

            print(f"*** {crnt_time} ***")
            if len(reblncng_time_list) == 0:  # For the first time
                # Set the initial portfolio valuation and store them.
                prtfl_expctd_valtn = pd.DataFrame([self._init_prtfl_valtn], index=[crnt_time], columns=["prtfl_expctd_valtn"])
                prtfl_obsrvd_valtn = pd.DataFrame([self._init_prtfl_valtn], index=[crnt_time], columns=["prtfl_obsrvd_valtn"])
                prtfl_valtn = pd.DataFrame([self._init_prtfl_valtn], index=[crnt_time], columns=["prtfl_valtn"])
                data_history["prtfl_expctd_valtn"] = pd.concat([data_history["prtfl_expctd_valtn"], prtfl_expctd_valtn], axis=0)
                data_history["prtfl_obsrvd_valtn"] = pd.concat([data_history["prtfl_obsrvd_valtn"], prtfl_obsrvd_valtn], axis=0)
                data_history["prtfl_valtn"] = pd.concat([data_history["prtfl_valtn"], prtfl_valtn], axis=0)
            else:  # For the other times
                # Calculate expected values and store them.
                # For the assets
                asset_expctd_valtns = pd.DataFrame(np.array(prev_asset_valtns) * np.array(1 + solver.asset_expctd_returns_), index=solver.asset_expctd_returns_.index, columns=solver.asset_expctd_returns_.columns)
                data_history["asset_expctd_returns"] = pd.concat([data_history["asset_expctd_returns"], solver.asset_expctd_returns_], axis=0)
                data_history["asset_expctd_risks"] = pd.concat([data_history["asset_expctd_risks"], solver.asset_expctd_risks_], axis=0)
                data_history["asset_expctd_valtns"] = pd.concat([data_history["asset_expctd_valtns"], asset_expctd_valtns], axis=0)
                # For the portfolio
                prtfl_expctd_valtn = pd.DataFrame(np.array(prev_prtfl_valtn) * np.array(1 + solver.prtfl_expctd_return_), index=solver.prtfl_expctd_return_.index, columns=["prtfl_expctd_valtn"])
                data_history["prtfl_expctd_return"] = pd.concat([data_history["prtfl_expctd_return"], solver.prtfl_expctd_return_], axis=0)
                data_history["prtfl_expctd_risk"] = pd.concat([data_history["prtfl_expctd_risk"], solver.prtfl_expctd_risk_], axis=0)
                data_history["prtfl_expctd_valtn"] = pd.concat([data_history["prtfl_expctd_valtn"], prtfl_expctd_valtn], axis=0)

                # Calculate the observed values and store them.
                # Common setting
                prev_crnt_prices = self._prices[reblncng_time_list[-1]:latest_time]
                kwargs = {"prices": prev_crnt_prices, "index": [crnt_time]}
                # For the assets
                asset_obsrvd_returns = calc_asset_obsrvd_returns(**kwargs)
                asset_obsrvd_risks = calc_asset_obsrvd_risks(**kwargs)
                asset_obsrvd_valtns = pd.DataFrame(np.array(prev_asset_valtns) * np.array(1 + asset_obsrvd_returns), index=[crnt_time], columns=prev_asset_valtns.columns)
                data_history["asset_obsrvd_returns"] = pd.concat([data_history["asset_obsrvd_returns"], asset_obsrvd_returns], axis=0)
                data_history["asset_obsrvd_risks"] = pd.concat([data_history["asset_obsrvd_risks"], asset_obsrvd_risks], axis=0)
                data_history["asset_obsrvd_valtns"] = pd.concat([data_history["asset_obsrvd_valtns"], asset_obsrvd_valtns], axis=0)
                # For the portfolio
                prtfl_obsrvd_return = calc_prtfl_obsrvd_return(asset_props=solver.asset_props_, columns=["prtfl_obsrvd_return"], **kwargs)
                prtfl_obsrvd_risk = calc_prtfl_obsrvd_risk(asset_props=solver.asset_props_, columns=["prtfl_obsrvd_risk"], **kwargs)
                prtfl_obsrvd_valtn = pd.DataFrame(np.array(prev_prtfl_valtn) * np.array(1 + prtfl_obsrvd_return), index=[crnt_time], columns=["prtfl_obsrvd_valtn"])
                data_history["prtfl_obsrvd_return"] = pd.concat([data_history["prtfl_obsrvd_return"], prtfl_obsrvd_return], axis=0)
                data_history["prtfl_obsrvd_risk"] = pd.concat([data_history["prtfl_obsrvd_risk"], prtfl_obsrvd_risk], axis=0)
                data_history["prtfl_obsrvd_valtn"] = pd.concat([data_history["prtfl_obsrvd_valtn"], prtfl_obsrvd_valtn], axis=0)

                # Calculate performance and store thme.
                # For the assets
                asset_returns = calc_asset_obsrvd_returns(prices=prev_crnt_prices.iloc[[0, -1], :], frequency=1, index=[crnt_time])
                asset_valtns = pd.DataFrame(np.array(prev_asset_valtns) * np.array(1 + asset_returns), index=[crnt_time], columns=prev_asset_valtns.columns)
                data_history["asset_returns"] = pd.concat([data_history["asset_returns"], asset_returns], axis=0)
                data_history["asset_valtns"] = pd.concat([data_history["asset_valtns"], asset_valtns], axis=0)
                # For the portfolio
                #prtfl_return = pd.DataFrame([(np.array(prev_asset_valtns) * np.array(asset_returns)).sum().sum() / prev_asset_valtns.sum().sum()], index=[crnt_time], columns=["prtfl_return"])
                prtfl_return = pd.DataFrame([(asset_valtns.sum().sum() - prev_asset_valtns.sum().sum()) / prev_asset_valtns.sum().sum()], index=[crnt_time], columns=["prtfl_return"])
                prtfl_valtn = pd.DataFrame(np.array(prev_prtfl_valtn) * np.array(1 + prtfl_return), index=[crnt_time], columns=prtfl_valtn.columns)
                data_history["prtfl_return"] = pd.concat([data_history["prtfl_return"], prtfl_return], axis=0)
                data_history["prtfl_valtn"] = pd.concat([data_history["prtfl_valtn"], prtfl_valtn], axis=0)

            # Define a problem at the current date-time and solve the problem.
            problem.define(crnt_prices, crnt_time)

            # Calculate the asset proportions and the asset valuations after rebalancing and store them.
            solver.solve(problem)
            asset_valtns_reblncd = prtfl_valtn.iloc[0, 0] * solver.asset_props_
            data_history["asset_props"] = pd.concat([data_history["asset_props"], solver.asset_props_], axis=0)
            data_history["asset_valtns_reblncd"] = pd.concat([data_history["asset_valtns_reblncd"], asset_valtns_reblncd], axis=0)

            # Store and back up some information for the next date-time.
            reblncng_time_list.append(crnt_time)
            prev_prices = crnt_prices.copy()
            prev_asset_valtns = asset_valtns_reblncd.copy()
            prev_prtfl_valtn = prtfl_valtn.copy()
            crnt_time += timedelta(days=self._min_reblncng_intrvl_day)

        # Classify historical data to expected value and observed value.
        # For expected value
        data_history["prtfl_expctd_value"] = pd.DataFrame()
        data_history["prtfl_expctd_value"]["valtn"] = data_history["prtfl_expctd_valtn"]["prtfl_expctd_valtn"]
        data_history["prtfl_expctd_value"]["return"] = data_history["prtfl_expctd_return"]["prtfl_expctd_return"]
        data_history["prtfl_expctd_value"]["lower"] = data_history["prtfl_expctd_return"]["prtfl_expctd_return"] - data_history["prtfl_expctd_risk"]["prtfl_expctd_risk"]
        data_history["prtfl_expctd_value"]["upper"] = data_history["prtfl_expctd_return"]["prtfl_expctd_return"] + data_history["prtfl_expctd_risk"]["prtfl_expctd_risk"]
        data_history["prtfl_expctd_value"]["risk"] = data_history["prtfl_expctd_risk"]["prtfl_expctd_risk"]
        data_history["prtfl_expctd_value"]["sharp_ratio"] = data_history["prtfl_expctd_return"]["prtfl_expctd_return"] / data_history["prtfl_expctd_risk"]["prtfl_expctd_risk"]
        # For observed value
        data_history["prtfl_obsrvd_value"] = pd.DataFrame()
        data_history["prtfl_obsrvd_value"]["valtn"] = data_history["prtfl_obsrvd_valtn"]["prtfl_obsrvd_valtn"]
        data_history["prtfl_obsrvd_value"]["return"] = data_history["prtfl_obsrvd_return"]["prtfl_obsrvd_return"]
        data_history["prtfl_obsrvd_value"]["lower"] = data_history["prtfl_obsrvd_return"]["prtfl_obsrvd_return"] - data_history["prtfl_obsrvd_risk"]["prtfl_obsrvd_risk"]
        data_history["prtfl_obsrvd_value"]["upper"] = data_history["prtfl_obsrvd_return"]["prtfl_obsrvd_return"] + data_history["prtfl_obsrvd_risk"]["prtfl_obsrvd_risk"]
        data_history["prtfl_obsrvd_value"]["risk"] = data_history["prtfl_obsrvd_risk"]["prtfl_obsrvd_risk"]
        data_history["prtfl_obsrvd_value"]["sharp_ratio"] = data_history["prtfl_obsrvd_return"]["prtfl_obsrvd_return"] / data_history["prtfl_obsrvd_risk"]["prtfl_obsrvd_risk"]
        # Match the date-time indices of the expected values to the ones of the observed values
        data_history["prtfl_expctd_value"].index = data_history["prtfl_obsrvd_value"].index

        # Save the historical data.
        # Directory preparing
        if not os.path.exists(f"{self._result_dir}"):
            os.makedirs(f"{self._result_dir}")
        # As dump file
        joblib.dump(data_history, os.path.join(self._result_dir, "data_history"))
        # As CSV file
        data_history["asset_expctd_returns"].to_csv(os.path.join(self._result_dir, "asset_expected_returns_history.csv"))
        data_history["asset_obsrvd_returns"].to_csv(os.path.join(self._result_dir, "asset_obsrvd_returns_history.csv"))
        data_history["asset_returns"].to_csv(os.path.join(self._result_dir, "asset_returns_history.csv"))
        data_history["prtfl_return"].to_csv(os.path.join(self._result_dir, "portfolio_return_history.csv"))
        data_history["asset_props"].to_csv(os.path.join(self._result_dir, "asset_proportion_history.csv"))
