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

    params : dict
        The other parameters of the trigger, the problem and the solver.
    """
    def __init__(self, prices, start_time, end_time, **params):
        self._prices = prices
        self._start_time = start_time
        self._end_time = end_time

    def execute(self):
        """Execute the simulation."""
        init_prtfl_valtn = 100.0
        window_day = 28
        min_reblncng_intrvl_day = 1

        params = {}
        params["reblncng_intrvl_day"] = 28
        params["return_lower_qntl"] = 0.7
        trigger = Trigger(RegularBasis(**params))
        problem = RiskMinimization(**params)
        solver = Solver(EqualProportion(**params))

        # Set objects for the simulation.
        prev_prices = None
        reblncng_time_list = []
        window = timedelta(days=window_day)
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
                crnt_time += timedelta(days=min_reblncng_intrvl_day)
                continue

            print(f"*** {crnt_time} ***")
            if len(reblncng_time_list) == 0:  # For the first time
                # Set the initial portfolio valuation and store them.
                prtfl_expctd_valtn = pd.DataFrame([init_prtfl_valtn], index=[crnt_time], columns=["prtfl_expctd_valtn"])
                prtfl_obsrvd_valtn = pd.DataFrame([init_prtfl_valtn], index=[crnt_time], columns=["prtfl_obsrvd_valtn"])
                prtfl_valtn = pd.DataFrame([init_prtfl_valtn], index=[crnt_time], columns=["prtfl_valtn"])
            else:  # For the other times
                # Calculate expected values and store them.
                # For the assets
                asset_expctd_valtns = pd.DataFrame(np.array(prev_asset_valtns) * np.array(1 + solver.asset_expctd_returns_), index=solver.asset_expctd_returns_.index, columns=solver.asset_expctd_returns_.columns)
                # For the portfolio
                prtfl_expctd_valtn = pd.DataFrame(np.array(prev_prtfl_valtn) * np.array(1 + solver.prtfl_expctd_return_), index=solver.prtfl_expctd_return_.index, columns=["prtfl_expctd_valtn"])

                # Calculate the observed values and store them.
                # Common setting
                prev_crnt_prices = self._prices[reblncng_time_list[-1]:latest_time]
                kwargs = {"prices": prev_crnt_prices, "index": [crnt_time]}
                # For the assets
                asset_obsrvd_returns = calc_asset_obsrvd_returns(**kwargs)
                asset_obsrvd_risks = calc_asset_obsrvd_risks(**kwargs)
                asset_obsrvd_valtns = pd.DataFrame(np.array(prev_asset_valtns) * np.array(1 + asset_obsrvd_returns), index=[crnt_time], columns=prev_asset_valtns.columns)
                # For the portfolio
                prtfl_obsrvd_return = calc_prtfl_obsrvd_return(asset_props=solver.asset_props_, columns=["prtfl_obsrvd_return"], **kwargs)
                prtfl_obsrvd_risk = calc_prtfl_obsrvd_risk(asset_props=solver.asset_props_, columns=["prtfl_obsrvd_risk"], **kwargs)
                prtfl_obsrvd_valtn = pd.DataFrame(np.array(prev_prtfl_valtn) * np.array(1 + prtfl_obsrvd_return), index=[crnt_time], columns=["prtfl_obsrvd_valtn"])

                # Calculate performance and store thme.
                # For the assets
                asset_returns = calc_asset_obsrvd_returns(prices=prev_crnt_prices.iloc[[0, -1], :], frequency=1, index=[crnt_time])
                asset_valtns = pd.DataFrame(np.array(prev_asset_valtns) * np.array(1 + asset_returns), index=[crnt_time], columns=prev_asset_valtns.columns)
                # For the portfolio
                #prtfl_return = pd.DataFrame([(np.array(prev_asset_valtns) * np.array(asset_returns)).sum().sum() / prev_asset_valtns.sum().sum()], index=[crnt_time], columns=["prtfl_return"])
                prtfl_return = pd.DataFrame([(asset_valtns.sum().sum() - prev_asset_valtns.sum().sum()) / prev_asset_valtns.sum().sum()], index=[crnt_time], columns=["prtfl_return"])
                prtfl_valtn = pd.DataFrame(np.array(prev_prtfl_valtn) * np.array(1 + prtfl_return), index=[crnt_time], columns=prtfl_valtn.columns)

            # Define a problem at the current date-time and solve the problem.
            problem.define(crnt_prices, crnt_time)

            # Calculate the asset proportions and the asset valuations after rebalancing and store them.
            solver.solve(problem)
            asset_valtns_reblncd = prtfl_valtn.iloc[0, 0] * solver.asset_props_

            # Store and back up some information for the next date-time.
            reblncng_time_list.append(crnt_time)
            prev_prices = crnt_prices.copy()
            prev_asset_valtns = asset_valtns_reblncd.copy()
            prev_prtfl_valtn = prtfl_valtn.copy()
            crnt_time += timedelta(days=min_reblncng_intrvl_day)
