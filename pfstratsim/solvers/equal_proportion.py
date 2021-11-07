import numpy as np
import pandas as pd
import time
from pyomo.environ import *

from .solver_interface import SolverInterface


class EqualProportion(SolverInterface):
    """The solver algorithm using equal proportion.

    This is a concrete class on the strategy pattern for solver algorithms.

    Parameters
    ----------
    params : dict
        The parameters not to be used in this class but necessary just to realize the API that can call the constructor
        of all the solver algorithm by one way.
    """
    def __init__(self, **params):
        pass

    def solve(self, problem, is_print=False, **params):
        """Solve the problem given.

        Parameters
        ----------
        problem : Problem
            The optimization problem to determine the asset proportions.

        is_print : bool, default False
            The option whether to output the result summary.

        params : dict
            The parameters not to be used in this class but necessary just to realize the API that can call this method
            of all the solver algorithm by one way.
        """
        model = problem.cncrt_model_

        start_time = time.perf_counter()

        # Calculate the asset proportions.
        for a in model.set_asset:
            model.asset_props[a] = 1.0 / len(model.asset_props)

        # Calculate the expected return of the portfolio
        prtfl_expctd_return = 0.0
        for a in model.set_asset:
            prtfl_expctd_return += value(model.asset_expctd_returns[a] * model.asset_props[a])
        model.prtfl_expctd_return = prtfl_expctd_return

        # Calculate the expected risk of the portfolio
        prtfl_expctd_var = 0.0
        for a in model.set_asset:
            for a1 in model.set_asset:
                prtfl_expctd_var += value(model.asset_expctd_corr_cf[a, a1] * model.asset_expctd_risks[a] * model.asset_expctd_risks[a1] * model.asset_props[a] * model.asset_props[a1])
        model.prtfl_expctd_risk = np.sqrt(prtfl_expctd_var)

        comp_time = time.perf_counter() - start_time
        if is_print:
            print(f'computation time = {comp_time}')
            print(f'portfolio risk = {value(model.prtfl_expctd_risk) * 100}[%]')
            print(f'portfolio return = {value(model.prtfl_expctd_return) * 100}[%]')
            print(f'sharp ratio = {value(model.prtfl_expctd_return) / value(model.portfolio_expctd_risk)}')

            for a, asset_name in enumerate(problem.asset_name_list):
                print(f'{asset_name}: {value(model.asset_props[a]) * 100}[%]')
