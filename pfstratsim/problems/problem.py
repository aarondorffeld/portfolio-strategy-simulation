from pyomo.environ import *

from ..utils import calc_corr_cf, calc_asset_expctd_returns, calc_asset_expctd_risks


class Problem(object):
    """The base class for all problems.

    Parameters
    ----------
    params : dict
        The parameters not to be used in this class but necessary just to realize the API that can call the constructor
        of all the problems by one way.

    Attributes
    ----------
    _abst_model_ : AbstractModel
        The abstract otpimization model for the problem without specific values defined.

    cncrt_model_ : ConcreteModel
        The concrete optimization model for the problem with specific values defined.

    asset_name_list_ : list of shape (num_assets) and str
        The list of the asset names.

    _num_assets_ : int
        The number of the assets.

    _asset_expctd_returns_ : Series of shape (num_assets) and float
        The expected returns of the assets.

    _asset_expctd_risks_ : Series of shape (num_assets) and float
        The expected risks of the assets.

    _asset_expctd_corr_cf_ : DataFrame of shape (num_assets, num_assets) and float
        The expected correlation coefficients of the assets.

    crnt_time_ : Timestamp
        The current date-time at which the problem is defined.
    """
    def __init__(self, **params):
        pass

    def _reset(self, prices, crnt_time):
        """Reset all the simulation parameters.

        Parameters
        ----------
        prices : DataFrame of shape (num_times, num_assets) and float
            The historical prices of the assets.

        crnt_time : Timestamp
            The current date-time at which the problem is defined.
        """
        self._abst_model_ = None
        self._cncrt_model_ = None
        self._asset_name_list_ = prices.columns
        self._num_assets_ = len(prices.columns)
        self._asset_expctd_returns_ = calc_asset_expctd_returns(prices=prices, dtype="Series")
        self._asset_expctd_risks_ = calc_asset_expctd_risks(prices=prices, dtype="Series")
        self._asset_expctd_corr_cf_ = calc_corr_cf(prices=prices)
        self._crnt_time_ = crnt_time

    def define(self, prices, crnt_time):
        """Define the problem with the parameters given.

        Parameters
        ----------
        prices : DataFrame of shape (num_times, num_assets) and float
            The historical prices of the assets.

        crnt_time : Timestamp
            The current date-time at which the problem is defined.
        """
        self._reset(prices, crnt_time)
        self._build_abst_model()
        self._build_cncrt_model()
        is_success = self._validate_model()
        return is_success

    def _build_abst_model(self):
        """Build the abstract optimization model for the problem."""
        model = AbstractModel()
        model.set_asset = RangeSet(0, self._num_assets_ - 1)
        model.asset_expctd_returns = Param(model.set_asset, mutable=True)
        model.asset_expctd_risks = Param(model.set_asset, mutable=True)
        model.asset_expctd_corr_cf = Param(model.set_asset, model.set_asset, mutable=True)

        self._abst_model_ = model

    def _build_cncrt_model(self):
        """Build the concrete optimization model based on the abstract one."""
        model = self._abst_model_.create_instance()

        # Set constants.
        for a in model.set_asset:
            model.asset_expctd_returns[a] = self._asset_expctd_returns_[a]
            model.asset_expctd_risks[a] = self._asset_expctd_risks_[a]
            for a1 in model.set_asset:
                model.asset_expctd_corr_cf[a, a1] = self._asset_expctd_corr_cf_.iloc[a, a1]

        # Set variables.
        model.asset_props = Var(model.set_asset, bounds=(0.0, 1.0))
        model.prtfl_expctd_return = Var(within=Reals)
        model.prtfl_expctd_risk = Var(within=NonNegativeReals)

        # Set constraints.
        model.constr_prtfl_expctd_return = ConstraintList()
        model.constr_prtfl_expctd_return.add(model.prtfl_expctd_return == sum(model.asset_expctd_returns[a] * model.asset_props[a] for a in model.set_asset))

        model.constr_sum_asset_props = ConstraintList()
        model.constr_sum_asset_props.add(sum(model.asset_props[a] for a in model.set_asset) == 1.0)

        self._cncrt_model_ = model

    def _validate_model(self):
        is_success = True
        return is_success

    @property
    def cncrt_model_(self):
        return self._cncrt_model_

    @property
    def crnt_time_(self):
        return self._crnt_time_

    @property
    def asset_name_list_(self):
        return self._asset_name_list_
