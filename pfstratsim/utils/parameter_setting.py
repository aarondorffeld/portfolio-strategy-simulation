import os
import pandas as pd
import configparser


def read_params(setting_file_dir=".", setting_file_name="."):
    """Read the parameters from the setting file.

    Parameters
    ----------
    setting_file_dir : str, default None, default "."
        The directory of the parameter setting file.

    setting_file_name : str, default None, default "."
        The name of the parameter setting file.

    Returns
    -------
    params : dict
        The parameters to be used in the simulation.
    """
    param_file = configparser.ConfigParser()
    param_file.read(os.path.join(setting_file_dir, setting_file_name), "utf-8")

    section_list = ["dataset", "simulation", "trigger", "problem", "solver"]
    dataset_param_set = {
        "asset_name_list": "csv_file",
        "start_time": pd.Timestamp,
        "end_time": pd.Timestamp,
        "interest_rate": float,
    }
    simulation_param_set = {
        "init_prtfl_valtn": float,
        "window_day": int,
        "min_reblncng_intrvl_day": int,
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
        "max_time_limit": int,
    }

    all_param_set = {
        "dataset": dataset_param_set,
        "simulation": simulation_param_set,
        "trigger": trigger_param_set,
        "problem": problem_param_set,
        "solver": solver_param_set,
    }

    params = {}
    for section, param_set in all_param_set.items():
        for param_name, param_type in param_set.items():
            try:
                param_value = param_file.get(section, param_name)
            except Exception as e:
                print(e)
                continue
            param_value = del_comment(param_value)
            if param_type == int:
                params[param_name] = int(param_value)
            elif param_type == float:
                params[param_name] = float(param_value)
            elif param_type == str:
                params[param_name] = str(param_value)
            elif param_type == pd.Timestamp:
                params[param_name] = pd.to_datetime(param_value)
            elif param_type == "csv_file":
                columns_name = param_name.replace("_list", "")
                params[param_name] = pd.read_csv(os.path.join(setting_file_dir, param_value))[columns_name].to_list()
            else:
                message = f"Invalid value for 'param_type': {param_type}"
                ValueError(message)

    return params


def del_comment(string):
    """Delete the comment from the parameter setting string.

    Parameters
    ----------
    string : str, default None
        The parameter setting string probably with the comment.

    Returns
    -------
    string : str
        The parameter setting string without the comment.
    """
    return string.split('#')[0].replace(' ', '')
