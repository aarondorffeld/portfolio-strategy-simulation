from datetime import timedelta

from .trigger_interface import TriggerInterface


class RegularBasis(TriggerInterface):
    """The trigger algorithm using regular basis.

    This is a concrete class on the strategy pattern for trigger algorithms.

    Parameters
    ----------
    reblncng_intrvl_day : int
        The interval for rebalancing.

    params : dict
        The parameters not to be used in this class but necessary just to realize the API that can call this method of
        all the trigger algorithms by one way.
    """
    def __init__(self, reblncng_intrvl_day, **params):
        self._reblncng_intrvl_day = reblncng_intrvl_day

    def assess(self, crnt_time, reblncng_time_list, **params):
        """Assess the necessity of rebalancing.

        Parameters
        ----------
        crnt_time : Timestamp
            The current date-time at which the necessity for rebalancing is assessed.

        reblncng_time_list : list of Timestamp
            The list of the date-times at which the rebalancings are performed.

        params : dict
            The parameters not to be used in this class but necessary just to realize the API that can call this method
            of all the trigger algorithms by one way.

        Returns
        -------
        is_reblncng : bool
            The necessity of rebalancing. If "True", it is necessary; else, it is unnecessary.

        None : None
            The object not to be used but necessary just for API consistency of the trigger algorithm classes.
        """
        if len(reblncng_time_list) > 0:
            is_reblncng = reblncng_time_list[-1] + timedelta(days=self._reblncng_intrvl_day) <= crnt_time
        else:
            is_reblncng = True
        return is_reblncng, None
