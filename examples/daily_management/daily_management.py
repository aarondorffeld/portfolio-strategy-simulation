import os
import shutil
import numpy as np
import pandas as pd
from datetime import timedelta
import configparser
import joblib

from pfstratsim.datasets import fetch_prices
from pfstratsim.problems import *
from pfstratsim.triggers import *
from pfstratsim.solvers import *


def main():
    # Read the parameters
    crnt_dir = os.path.dirname(os.path.normpath(__file__))
    param_file_name = "parameters.ini"
    params = read_params(crnt_dir, param_file_name)

    is_update = params.get("is_update")
    price_file_name = params.get("price_file_name")
    price_start_time = params.get("price_start_time")
    crnt_time = params.get("crnt_time")
    window_day = params.get("window_day")
    prev_time = params.get("prev_time")
    prtfl_valtn = params.get("prtfl_valtn")

    # Set the input and output directories
    input_dir = os.path.join(crnt_dir, str(prev_time).split(" ")[0])
    output_dir = os.path.join(crnt_dir, str(crnt_time).split(" ")[0])
    objects_dir = os.path.join(output_dir, "objects")
    if not os.path.exists(objects_dir):
        os.makedirs(objects_dir)

    # Prepare for updating the prices data
    prices = pd.read_csv(os.path.join(crnt_dir, price_file_name), index_col=0)
    prices.index = pd.to_datetime(prices.index)
    asset_name_list = prices.columns.to_list()

    # Update the price data
    if is_update:
        if price_start_time is None:
            price_start_time = prices.index[-1]
        else:
            prices = pd.DataFrame()
        start_time = price_start_time
        if start_time <= crnt_time:
            prices_added = fetch_prices(
                asset_name_list=asset_name_list,
                start_time=start_time,
                end_time=crnt_time,
                is_save_each=False,
                is_save_all=False,
                save_dir=crnt_dir,
                **params
            )
            prices = pd.concat([prices, prices_added], axis=0).groupby(level=0).last()
            prices.to_csv(os.path.join(crnt_dir, price_file_name))

    # Prepare for assessing the necessity of rebalancing and calculating the optimal asset proportions
    start_time = crnt_time - timedelta(days=window_day)
    crnt_prices = prices[start_time:crnt_time]
    if prev_time is None:
        prev_prices = None
    else:
        prev_prices = pd.read_csv(os.path.join(input_dir, "crnt_prices.csv"), index_col=0)
        prev_prices.index = pd.to_datetime(prev_prices.index)

    trigger = Trigger(IdenticalDistributionTest(**params))
    problem = SharpeRatioMaximization(**params)
    solver = Solver(MathematicalProgramming(**params))

    # Asses the necessity of rebalancing
    is_reblncng, idntcl_dstrbtn_prob = trigger.assess(
        crnt_time=crnt_time,
        crnt_prices=crnt_prices,
        prev_prices=prev_prices,
    )
    if idntcl_dstrbtn_prob is None:
        idntcl_dstrbtn_prob = pd.DataFrame([np.nan for _ in range(len(asset_name_list))], index=asset_name_list).T
    idntcl_dstrbtn_prob.index = ['idntcl_dstrbtn_prob']
    summary = idntcl_dstrbtn_prob.copy()

    # Store the main data
    shutil.copy2(os.path.join(crnt_dir, param_file_name), output_dir)
    crnt_prices.to_csv(os.path.join(output_dir, "crnt_prices.csv"))
    joblib.dump(params, os.path.join(objects_dir, "params"))
    joblib.dump(prtfl_valtn, os.path.join(objects_dir, "prtfl_valtn"))
    joblib.dump(prev_prices, os.path.join(objects_dir, "prev_prices"))
    joblib.dump(crnt_time, os.path.join(objects_dir, "crnt_time"))
    joblib.dump(crnt_prices, os.path.join(objects_dir, "crnt_prices"))
    joblib.dump(trigger, os.path.join(objects_dir, "trigger"))
    joblib.dump(idntcl_dstrbtn_prob, os.path.join(objects_dir, "idntcl_dstrbtn_prob"))

    # Calculate the optimal asset proportions
    if is_reblncng:
        is_success = problem.define(crnt_prices, crnt_time)
        joblib.dump(problem, os.path.join(objects_dir, "problem"))
        if is_success:
            is_success = solver.solve(problem, **params)
            joblib.dump(solver, os.path.join(objects_dir, "solver"))
            if is_success:
                asset_props = solver.asset_props_
                asset_props.index = ['asset_props']
                asset_valtns_reblncd = prtfl_valtn * asset_props
                asset_valtns_reblncd.index = ['asset_valtns_reblncd']
                latest_prices = pd.DataFrame(crnt_prices.iloc[-1,:]).T
                latest_prices.index = ['latest_prices']
                asset_amounts = pd.DataFrame(asset_valtns_reblncd.values / latest_prices.values, index=['asset_amounts'], columns=asset_name_list)
                summary = pd.concat([summary, asset_props, asset_valtns_reblncd, latest_prices, asset_amounts], axis=0)

                # Store the main data
                joblib.dump(asset_props, os.path.join(objects_dir, "asset_props"))
                joblib.dump(asset_valtns_reblncd, os.path.join(objects_dir, "asset_valtns_reblncd"))
                joblib.dump(latest_prices, os.path.join(objects_dir, "latest_prices"))
                joblib.dump(asset_amounts, os.path.join(objects_dir, "asset_amounts"))
                joblib.dump(summary, os.path.join(objects_dir, "summary"))
            else:
                print("Problem solving failed.")
        else:
            print("Problem defining failed.")
    else:
        print("Rebalancing not necessary.")

    # Store the main data
    summary.to_csv(os.path.join(output_dir, "summary.csv"))


def read_params(setting_file_dir=".", setting_file_name="."):
    """Read the parameters from the setting file."""
    param_file = configparser.ConfigParser()
    param_file.read(os.path.join(setting_file_dir, setting_file_name), "utf-8")

    dataset_param_set = {
        "is_update": bool,
        "price_file_name": str,
        "price_start_time": pd.Timestamp,
        "prev_time": pd.Timestamp,
        "crnt_time": pd.Timestamp,
        "window_day": int,
        "interest_rate": float,
    }
    trigger_param_set = {
        "trigger_class": str,
        "test_method": str,
        "prob_thrshld": float,
        "reblncng_intrvl_day": int,
    }
    problem_param_set = {
        "problem_class": str,
        "return_lower_qntl": float,
    }
    solver_param_set = {
        "solver_class": str,
        "solver_name": str,
        "is_print": bool,
        "tee": bool,
        "max_time_limit": int,
    }
    other_param_set = {
        "prtfl_valtn": float,
    }

    all_param_set = {
        "dataset": dataset_param_set,
        "trigger": trigger_param_set,
        "problem": problem_param_set,
        "solver": solver_param_set,
        "other": other_param_set,
    }

    params = {}
    for section, param_set in all_param_set.items():
        for param_name, param_type in param_set.items():
            try:
                param_value = param_file.get(section, param_name)
            except Exception as e:
                print(e)
                continue
            param_value = param_value.split('#')[0].replace(' ', '')
            if param_type == int:
                params[param_name] = int(param_value)
            elif param_type == float:
                params[param_name] = float(param_value)
            elif param_type == bool:
                params[param_name] = param_value == "True"
            elif param_type == str:
                params[param_name] = str(param_value)
            elif param_type == pd.Timestamp:
                params[param_name] = pd.to_datetime(param_value)
            elif param_type == "csv_file":
                columns_name = param_name.replace("_list", "")
                try:
                    params[param_name] = pd.read_csv(os.path.join(setting_file_dir, param_value))[columns_name].to_list()
                except:
                    params[param_name] = None
            else:
                message = f"Invalid value for 'param_type': {param_type}"
                ValueError(message)

    return params


if __name__ == "__main__":
    main()