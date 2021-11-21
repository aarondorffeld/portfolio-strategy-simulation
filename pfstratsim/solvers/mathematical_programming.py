import pandas as pd
import time
from pyomo.environ import *

from .solver_interface import SolverInterface


class MathematicalProgramming(SolverInterface):
    """The solver algorithm using mathematical programming.

    This is a concrete class on the strategy pattern for solver algorithms.

    Parameters
    ----------
    solver_name : {"baron", "gurobi"}
        The name of the optimization solver to solve the problem formulated as mathematical programming problem.

    params : dict
        The parameters not to be used in this class but necessary just to realize the API that can call the constructor
        of all the solver algorithm by one way.
    """
    def __init__(self, solver_name=None, **params):
        self._solver_name = solver_name

    def solve(self, problem, is_print=False, tee=False, max_time_limit=-1, **params):
        """Solve the problem given.

        Parameters
        ----------
        problem : Problem
            The optimization problem to determine the asset proportions.

        is_print : bool, default False
            The option whether to output the result summary.

        tee : bool, default False
            The option whether to output the progress of the optimization process.

        max_time_limit : int, default=-1
            The time to abort the solution process.

        params : dict
            The parameters not to be used in this class but necessary just to realize the API that can call this method
            of all the solver algorithm by one way.

        Returns
        -------
        is_success : bool
            True if the problem is solved properly.
        """
        model = problem.cncrt_model_
        opt = SolverFactory(self._solver_name)

        start_time = time.perf_counter()
        result = opt.solve(model, tee=tee, options={'Maxtime': max_time_limit})#, 'NumSol':100}, keepfiles=True)
        comp_time = time.perf_counter() - start_time
        if result.solver.termination_condition == TerminationCondition.infeasible:
            print('The problem infeasible.')
            is_success = False
        elif result.solver.termination_condition == 'maxTimeLimit':
            print('Max time limit reached.')
            is_success = False
        else:
            is_success = True

        if is_print:
            print(f'computation time = {comp_time}')
            print(f'objective = {value(model.objctv)}')
            print(f'portfolio risk = {value(model.prtfl_expctd_risk) * 100}[%]')
            print(f'portfolio return = {value(model.prtfl_expctd_return) * 100}[%]')
            print(f'sharpe ratio = {value(model.prtfl_expctd_return) / value(model.prtfl_expctd_risk)}')

            for a, asset_name in enumerate(problem.asset_name_list_):
                print(f'{asset_name}: {value(model.asset_props[a]) * 100}[%]')

        return is_success
