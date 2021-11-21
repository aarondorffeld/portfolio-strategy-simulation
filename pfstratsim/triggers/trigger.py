class Trigger(object):
    """The trigger to determine the rebalancing timings.

    This is the context class on the strategy pattern for trigger algorithms.

    Parameters
    ----------
    cncrt_trigger : derived class from TriggerInterface
        The trigger algorithm to determine the rebalancing timings, whose class is the concrete class on the strategy pattern.
    """
    def __init__(self, cncrt_trigger):
        self._cncrt_trigger = cncrt_trigger

    def assess(self, **params):
        """Assess the necessity of rebalancing.

        Parameters
        ----------
        params : dict
            The parameters of the trigger algorithm and the ones not to be used in this class but necessary just to
            realize the API that can call this method of all the trigger algorithms by one way.

        Returns
        -------
        (depending on the trigger algorithm used.)
        """
        return self._cncrt_trigger.assess(**params)
