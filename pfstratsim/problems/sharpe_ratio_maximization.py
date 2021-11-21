from pyomo.environ import *
import warnings

from .problem import Problem


class SharpeRatioMaximization(Problem):
    """Problem to find a solution that maximizes the portfolio Sharpe ratio.

    Parameters
    ----------
    params : dict
        The parameters not to be used in this class but necessary just to realize the API that can call the constructor
        of all the problem by one way.
    """
    def __init__(self, **params):
        super().__init__(**params)

    def _build_abst_model(self):
        """Build the abstract optimization model for the problem."""
        super()._build_abst_model()

    def _build_cncrt_model(self):
        """Build the concrete optimization model based on the abstract one."""
        super()._build_cncrt_model()
        model = self._cncrt_model_

        # Set an objective.
        expr = model.prtfl_expctd_return / model.prtfl_expctd_risk
        sense = maximize
        model.objctv = Objective(expr=expr, sense=sense)

        # Set Constraints.s
        model.constr_prtfl_expctd_risk = ConstraintList()
        model.constr_prtfl_expctd_risk.add(model.prtfl_expctd_risk ** 2 >=
            sum((model.asset_expctd_corr_cf[a, a1] * model.asset_expctd_risks[a] * model.asset_expctd_risks[a1] * model.asset_props[a] * model.asset_props[a1])
            for a in model.set_asset for a1 in model.set_asset))

        self._cncrt_model_ = model

    def _validate_model(self):
        """Validate the model if it is defined properly or not.

        Returns
        -------
        is_success : bool
            True if the problem is defined properly.
        """
        is_success = super()._validate_model()
        if 0.0 in self._asset_expctd_risks_.values:
            message = "Problem defining failed due to at least one asset with a zero risk which has an infinite " \
                      "Sharpe ratio. 'CASH' included in the assets causes this issue in most cases."
            warnings.warn(message)
            is_success = False
        return is_success
