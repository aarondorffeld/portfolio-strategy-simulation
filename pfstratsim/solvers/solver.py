import pandas as pd
from pyomo.environ import *


class Solver(object):
    """The solver to determine the asset proportions.

    This is the context class on the strategy pattern for solver algorithms.

    Parameters
    ----------
    cncrt_solver : derived class from SolverInterface
        The solver algorithm to determine the asset proportions, whose class is the concrete class on the strategy pattern.

    Attributes
    ----------
    asset_props_ : DataFrame of shape (num_times, num_assets) and float
        The asset proportions calculated by the solver algorithm.

    asset_expctd_returns_ : DataFrame of shape (num_times, num_assets) and float
        The expected returns of the assets on the problem.

    asset_expctd_risks_ : DataFrame of shape (num_times, num_assets) and float
        The expected risks of the assets on the problem.

    prtfl_expctd_return_ : DataFrame of shape (num_times=1, num_prtfls=1) and float
        The expected return of the portfolio based on the asset proportions.

    prtfl_expctd_risk_ : DataFrame of shape (num_times=1, num_prtfls=1) and float
        The expected risk of the portfolio based on the asset proportions.

    """
    def __init__(self, cncrt_solver):
        self._cncrt_solver = cncrt_solver

    def solve(self, problem, **params):
        """Solve the problem given.

        Parameters
        ----------
        problem : Problem
            The optimization problem to determine the asset proportions.

        params : dict
            The other parameters of the solver algorithm.

        Returns
        -------
        is_success : bool
            True if the problem is solved properly.
        """
        is_success = self._cncrt_solver.solve(problem, **params)
        if is_success:
            model = problem.cncrt_model_
            self._asset_props_ = pd.DataFrame(
                [value(model.asset_props[a]) for a in range(len(problem.asset_name_list_))],
                index=problem.asset_name_list_, columns=[problem.crnt_time_]
            ).T
            self._asset_expctd_returns_ = pd.DataFrame(
                [value(model.asset_expctd_returns[a]) for a in range(len(problem.asset_name_list_))],
                index=problem.asset_name_list_, columns=[problem.crnt_time_]
            ).T
            self._asset_expctd_risks_ = pd.DataFrame(
                [value(model.asset_expctd_risks[a]) for a in range(len(problem.asset_name_list_))],
                index=problem.asset_name_list_, columns=[problem.crnt_time_]
            ).T
            self._prtfl_expctd_return_ = pd.DataFrame(
                [value(model.prtfl_expctd_return)], index=[problem.crnt_time_], columns=['prtfl_expctd_return']
            )
            self._prtfl_expctd_risk_ = pd.DataFrame(
                [value(model.prtfl_expctd_risk)], index=[problem.crnt_time_], columns=['prtfl_expctd_risk']
            )
        return is_success

    @property
    def asset_props_(self):
        return self._asset_props_

    @property
    def asset_expctd_returns_(self):
        return self._asset_expctd_returns_

    @property
    def asset_expctd_risks_(self):
        return self._asset_expctd_risks_

    @property
    def prtfl_expctd_return_(self):
        return self._prtfl_expctd_return_

    @property
    def prtfl_expctd_risk_(self):
        return self._prtfl_expctd_risk_
