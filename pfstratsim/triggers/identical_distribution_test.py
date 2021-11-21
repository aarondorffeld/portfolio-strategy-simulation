import pandas as pd
from scipy import stats

from .trigger_interface import TriggerInterface
from ..utils import calc_asset_returns


class IdenticalDistributionTest(TriggerInterface):
    """The trigger algorithm using identical distribution test.

    This is a concrete class on the strategy pattern for trigger algorithms.

    Parameters
    ----------
    _test_method : {"anderson_darling", "kolmogorov_smirnov"}, default "anderson_darling"
        The statistical test method to test whether the two datasets of the returns calculated from the current and
        previous datasets of the prices come from an identical distribution.

    _prob_thrshld : float, default 0.05
        The threshold of the probability of identical distribution for the two datasets of the returns calculated from
        the current and previous datasets of the prices. If the probability is less than or equal to this threshold, the
        datasets are not regarded to come from an identical distribution.
    """
    def __init__(self, test_method="anderson_darling", prob_thrshld=0.05, **params):
        self._test_method = test_method
        self._prob_thrshld = prob_thrshld

    def assess(self, crnt_time, crnt_prices, prev_prices=None, **params):
        """Assess the necessity of rebalancing.

        Parameters
        ----------
        crnt_time : Timestamp
            The current date-time at which the necessity for rebalancing is assessed.

        crnt_prices : DataFrame of shape (num_times, num_assets) and float
            The current dataset of the prices used to determine the necessity of rebalancing.

        prev_prices : DataFrame of shape (num_times, num_assets) and float, default None
            The previous dataset of the prices used to calculate the expected values of return, risks and the like at
            the previous rebalancing time.

        params : dict
            The parameters not to be used in this class but necessary just to realize the API that can call this method
            of all the trigger algorithms by one way.

        Returns
        -------
        is_reblncng : bool
            The necessity of rebalancing. If "True", it is necessary; else, it is unnecessary.

        idntcl_dstrbtn_prob : DataFrame of shape (num_times=1, num_assets) and float
            The probability of identical distribution that the two datasets of the returns calculated from the current
            and previous datasets of the prices come from an identical distribution.
        """
        if prev_prices is not None:
            crnt_returns = calc_asset_returns(crnt_prices)
            prev_returns = calc_asset_returns(prev_prices)
            idntcl_dstrbtn_prob = pd.DataFrame()
            for asset_name in crnt_returns:
                # Anderson Darling Test for k-samples
                if self._test_method == "anderson_darling":
                    if asset_name == 'CASH':
                        p_value = 0.25  # the maximum value of the stats.anderson_ksamp return is 0.25
                    else:
                        p_value = stats.anderson_ksamp([crnt_returns[asset_name], prev_returns[asset_name]])[2]
                # Kolmogorov Smirnov Test
                elif self._test_method == "kolmogorov_smirnov":
                    p_value = stats.kstest(crnt_returns[asset_name], prev_returns[asset_name])[1]
                else:
                    p_value = None
                idntcl_dstrbtn_prob.loc[crnt_time, asset_name] = p_value
            is_reblncng = idntcl_dstrbtn_prob.min().min() <= self._prob_thrshld
        else:
            is_reblncng = True
            idntcl_dstrbtn_prob = None

        return is_reblncng, idntcl_dstrbtn_prob