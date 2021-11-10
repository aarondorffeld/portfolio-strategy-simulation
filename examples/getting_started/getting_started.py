import os
import pandas as pd

from pfstratsim.simulations import Simulation
from pfstratsim.datasets import load_sample_prices
from pfstratsim.utils import plot

def main():
    """Load the sample prices and execute a simulation"""
    start_time = pd.to_datetime("2019-08-01")
    end_time = pd.to_datetime("2021-08-01")

    prices = load_sample_prices(
        asset_name_list=["ASSET0", "ASSET1", "ASSET2", "ASSET3"],
        start_time=start_time,
        end_time=end_time,
        interest_rate=0.0001,
    )

    solver_class = "mathematical_programming"
    if solver_class == "equal_proportion":
        solver_class_abrvtd = "ep"
    elif solver_class == "mathematical_programming":
        solver_class_abrvtd = "mp"
    else:
        message = f"Invalid value for 'self._solver_class': {solver_class}." \
                  f"'self._solver_class' must be in ['mathematical_programming', 'equal_proportion']."
        raise ValueError(message)

    result_dir = os.path.join(os.path.dirname(os.path.normpath(__file__)), "results", f"risk_{solver_class_abrvtd}")

    sim = Simulation(
        trigger_class="identical_distribution_test",
        problem_class="risk_minimization",
        solver_class=solver_class,
        prices=prices,
        start_time=start_time,
        end_time=end_time,
        result_dir=result_dir,
        return_lower_qntl=0.7,
        reblncng_intrvl_day=28,
        solver_name="baron",
        max_time_limit=2,
    )
    sim.execute()

    plot(result_dir, result_dir)


if __name__ == "__main__":
    main()
