import os
import pandas as pd

from pfstratsim.simulations import Simulation
from pfstratsim.datasets import load_sample_prices
from pfstratsim.utils import plot, read_params

def main():
    """Read the parameters, load the sample prices and execute a simulation"""
    crnt_dir = os.path.dirname(os.path.normpath(__file__))
    params = read_params(crnt_dir, "parameters.ini")

    prices = load_sample_prices(**params)

    problem_class = params["problem_class"]
    if problem_class == "risk_minimization":
        problem_class_abrvtd = "risk"
    elif problem_class == "sharpe_ratio_maximization":
        problem_class_abrvtd = "sr"
    else:
        message = f"Invalid value for 'problem_class': {problem_class}." \
                  f"'problem_class' must be in ['risk_minimization', 'sharpe_ratio_maximization']."
        raise ValueError(message)

    solver_class = params["solver_class"]
    if solver_class == "equal_proportion":
        solver_class_abrvtd = "ep"
    elif solver_class == "mathematical_programming":
        solver_class_abrvtd = "mp"
    else:
        message = f"Invalid value for 'solver_class': {solver_class}." \
                  f"'solver_class' must be in ['equal_proportion', 'mathematical_programming']."
        raise ValueError(message)

    result_dir = os.path.join(os.path.dirname(os.path.normpath(__file__)), "results", f"{problem_class_abrvtd}_{solver_class_abrvtd}")

    sim = Simulation(prices=prices, result_dir=result_dir, **params)
    sim.execute()

    plot(result_dir, result_dir)


if __name__ == "__main__":
    main()
