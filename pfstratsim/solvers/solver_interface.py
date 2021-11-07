from abc import ABCMeta, abstractmethod


class SolverInterface(metaclass=ABCMeta):
    """The solver algorithm interface to be derived.

    This is the strategy class on the strategy pattern for solver algorithms.
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def solve(self, problem):
        pass
