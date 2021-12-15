import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import joblib


def plot(input_dir=".", output_dir="."):
    """Plot the historical data and output the figures into the output directory.

    Parameters
    ----------
    input_dir : str, default None, default "."
        The directory of the historical data.

    output_dir : str, default None, default "."
        The directory of the figures.
    """
    # Set the configures for plotting.
    plt.style.use("seaborn-whitegrid")
    plt.rcParams["font.size"] = 16

    # Extract the data to be plotted.
    data_history = joblib.load(os.path.join(input_dir, "data_history"))
    prices = data_history["prices"]
    asset_name_list = prices.columns
    idntcl_dstrbtn_prob_history = data_history["idntcl_dstrbtn_prob"]
    prtfl_expctd_value_history = data_history["prtfl_expctd_value"]
    prtfl_obsrvd_value_history = data_history["prtfl_obsrvd_value"]
    asset_returns_history = data_history["asset_returns"]
    asset_valtns_history = data_history["asset_valtns"]
    prtfl_valtn_history = data_history["prtfl_valtn"]
    asset_props_history = data_history["asset_props"]

    # Set the common setting.
    nrows = 10
    fig, ax = plt.subplots(nrows=nrows, figsize=(20, 5 * nrows), sharex="col")
    args_expctd = {"marker": "o", "label": "expected", "color": "green", "alpha": 0.3}
    args_obsrvd = {"marker": "o", "label": "observed", "color": "blue"}
    args_prfmnc = {"marker": "o", "label": "performance", "color": "blue"}  # , "linewidth": 3}
    suptitle = os.path.basename(input_dir)
    fig.suptitle(suptitle)

    # Plot the data.
    i = 0
    ax[i].set_title("The Prices")
    for asset in prices:
        ax[i].plot(prices[asset], label=asset)

    i += 1
    ax[i].set_title("The Identical Distribution Probabilities")
    for asset in idntcl_dstrbtn_prob_history:
        ax[i].plot(idntcl_dstrbtn_prob_history[asset], label=asset)

    i += 1
    ax[i].set_title("The Asset Proportions")
    bottom = np.zeros(len(asset_props_history))
    for a, asset_name in enumerate(asset_name_list):
        ax[i].bar(asset_props_history.index, asset_props_history[asset_name], bottom=bottom, width=1.0,
                  label=asset_name)
        bottom += np.array(asset_props_history.iloc[:, a])

    i += 1
    ax[i].set_title("The Asset Valuations")
    bottom = np.zeros(len(asset_valtns_history))
    for a, asset_name in enumerate(asset_name_list):
        ax[i].bar(asset_valtns_history.index, asset_valtns_history[asset_name], bottom=bottom, width=1.0,
                  label=asset_name)
        bottom += np.array(asset_valtns_history.iloc[:, a])

    i += 1
    ax[i].set_title("The Asset Returns")
    for asset in prices:
        ax[i].plot(asset_returns_history[asset], label=asset)

    i += 1
    ax[i].set_title("The Portfolio Valuation")
    ax[i].plot(prtfl_valtn_history["prtfl_valtn"], **args_prfmnc)

    i += 1
    ax[i].set_title("The Portfolio Expected/Observed Valuation")
    ax[i].plot(prtfl_expctd_value_history["valtn"], **args_expctd)
    ax[i].plot(prtfl_obsrvd_value_history["valtn"], **args_obsrvd)

    i += 1
    ax[i].set_title("The Portfolio Expected/Observed Sharpe Ratio")
    ax[i].plot(prtfl_expctd_value_history["sharpe_ratio"], **args_expctd)
    ax[i].plot(prtfl_obsrvd_value_history["sharpe_ratio"], **args_obsrvd)

    i += 1
    ax[i].set_title("The Portfolio Expected/Observed Return")
    ax[i].plot(prtfl_expctd_value_history["return"], **args_expctd)
    ax[i].plot(prtfl_expctd_value_history["lower"], **args_expctd, linestyle="--")
    ax[i].plot(prtfl_expctd_value_history["upper"], **args_expctd, linestyle="--")
    ax[i].plot(prtfl_obsrvd_value_history["return"], **args_obsrvd)
    ax[i].plot(prtfl_obsrvd_value_history["lower"], **args_obsrvd, linestyle="--")
    ax[i].plot(prtfl_obsrvd_value_history["upper"], **args_obsrvd, linestyle="--")

    i += 1
    ax[i].set_title("The Portfolio Expected/Observed Risk")
    ax[i].plot(prtfl_expctd_value_history["risk"], **args_expctd)
    ax[i].plot(prtfl_obsrvd_value_history["risk"], **args_obsrvd)

    for i in range(nrows):
        ax[i].tick_params(labelbottom=True)
        ax[i].legend()
        ax[i].legend(loc="center right")
    x_lower = prices.index[0]
    x_upper = prices.index[-1]
    x_delta = x_upper - x_lower
    ax[i].set_xlim((x_lower, x_upper + 0.2 * x_delta))

    # Save the figures.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdf = PdfPages(os.path.join(output_dir, f"{suptitle}_summary.pdf"))
    pdf.savefig()
    pdf.close()
    plt.savefig(os.path.join(output_dir, f"{suptitle}_summary.png"))
    plt.close()
