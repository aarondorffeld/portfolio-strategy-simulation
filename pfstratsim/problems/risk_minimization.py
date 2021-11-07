import numpy as np
from pyomo.environ import *

from .problem import Problem


class RiskMinimization(Problem):
    """Problem to find a solution that minimizes the portfolio risk.

    Parameters
    ----------
    return_lower_qntl : float
        The quantile value for the expected returns of the assets to determine the lower bound to be imposed to the
        expected return of the portfolio.

    params : dict
        The parameters not to be used in this class but necessary just to realize the API that can call the constructor
        of all the problems by one way.
    """
    def __init__(self, return_lower_qntl, **params):
        super().__init__(**params)
        self._return_lower_qntl = return_lower_qntl

    def _build_abst_model(self):
        """Build the abstract optimization model for the problem."""
        super()._build_abst_model()
        self._abst_model_.prtfl_expctd_return_lower = Param(within=Reals, mutable=True)

    def _build_cncrt_model(self):
        """Build the concrete optimization model based on the abstract one."""
        super()._build_cncrt_model()
        model = self._cncrt_model_

        # Set constants.
        model.prtfl_expctd_return_lower = np.quantile([value(model.asset_expctd_returns[a]) for a in model.set_asset], self._return_lower_qntl)

        # Set an objective.
        expr = model.prtfl_expctd_risk
        sense = minimize
        model.objctv = Objective(expr=expr, sense=sense)

        # Set constraints.
        model.constr_min_prtfl_expctd_return = ConstraintList()
        model.constr_min_prtfl_expctd_return.add(model.prtfl_expctd_return >= model.prtfl_expctd_return_lower)

        model.constr_prtfl_expctd_risk = ConstraintList()
        model.constr_prtfl_expctd_risk.add(model.prtfl_expctd_risk ** 2 >=
            sum((model.asset_expctd_corr_cf[a, a1] * model.asset_expctd_risks[a] * model.asset_expctd_risks[a1] * model.asset_props[a] * model.asset_props[a1])
            for a in model.set_asset for a1 in model.set_asset))

        self._cncrt_model_ = model
