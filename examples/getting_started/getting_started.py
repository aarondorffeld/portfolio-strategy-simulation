import pandas as pd

from pfstratsim.simulations import Simulation
from pfstratsim.datasets import load_sample_prices

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
    print(prices)
    
    sim = Simulation(
        prices=prices,
        start_time=start_time,
        end_time=end_time,
    )
    sim.execute()


if __name__ == "__main__":
    main()
