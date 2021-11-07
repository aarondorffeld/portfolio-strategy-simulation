from abc import ABCMeta, abstractmethod


class TriggerInterface(metaclass=ABCMeta):
    """Trigger algorithm interface to be derived.

    This is the strategy class on the strategy pattern for trigger algorithms.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def assess(self):
        pass
