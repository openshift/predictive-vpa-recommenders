import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, os, pickle, argparse, multiprocessing, logging
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import yaml
import recommender_config

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', UserWarning)


from recommender.functions import Theta_forecast, Theta_forecast_sktime, Naive_forecast, lr_forecast, autoarima_forecast, KNN_forecast, DT_forecast, VPA_forecast
from recommender.functions import perform_tests


def pando_recommender(y_segment, tree, window, limit = 5):
    forecasters = {'theta':Theta_forecast, 'theta-sktime': Theta_forecast_sktime, 'naive': Naive_forecast, 'linear': lr_forecast,'arima': autoarima_forecast, 'kn': KNN_forecast, 'dt': DT_forecast, "vpa": VPA_forecast}
    # Check y_segment length
    if len(y_segment) < window:
        forecast = np.zeros(window)
        forecast[-len(y_segment):] = y_segment
        prov = np.percentile(forecast, recommender_config.TARGET_PERCENTILE)
        return forecast, prov, "warmup"

    # get label for segment
    tests = perform_tests(y_segment, recommender_config.STAT_THRESHOLD, recommender_config.THETA_THRESHOLD, recommender_config.MAX_CHANGEPOINTS)
    label = int("".join(str(int(i)) for i in tests.values()), 2)
    print("Trace Behavior Label: {}".format(label))

    # get forecaster
    rec_name = tree[label]
    if type(rec_name) == float and np.isnan(rec_name):
        logging.debug("Unseen label: {}. Proceed to apply default VPA".format(label))
        rec_name = "vpa"
    forecaster = forecasters[rec_name]
    print("Trace Forecaster Selected: {}".format(rec_name))

    try:
        logging.info("Detected label: {}. Forecasting with: {}".format(label, rec_name))
        forecast = forecaster(y_segment, window, output=recommender_config.OUTPUT)
    except:
        logging.warning("Forecast is invalid, proceed to recommend previous usage.")
        forecast = y_segment[-window:]

    if max(forecast) > recommender_config.LIMIT * max(y_segment):
        logging.warning("Forecast is out of limits, proceed to recommend previous usage.")
        forecast = y_segment[-window:]
        prov = np.percentile(y_segment, recommender_config.TARGET_PERCENTILE)
    else:
        prov = np.percentile(forecast, recommender_config.TARGET_PERCENTILE)

    print("Forecasts: {}".format(forecast))
    print("Provision: {}".format(prov))
    return forecast, prov, label

def get_all_recommendation(y, tree, window = 20, sight = 240, default_usage=1000):
    start, end = 0, window
    N = len(y)

    prov, forecast = np.zeros(N), np.zeros(N)
    labels = []

    prov[start:end] = default_usage
    forecast[start:end] = default_usage

    while end < N:
        logging.info("Current segment {}-{}".format(start, end))
        forecast_start = end
        forecast_end = min(end + window, N)

        prev_usage = y[start:end]
        if end < sight:
            logging.info("Warmup phase")
            prov[forecast_start:forecast_end] = np.percentile(prev_usage, recommender_config.TARGET_PERCENTILE)
            forecast[forecast_start:forecast_end] = prev_usage
        else:
            prev_sight = y[end - sight:end]
            forecast_segment, rec, label = pando_recommender(prev_sight, tree, forecast_end - forecast_start)
            forecast[forecast_start:forecast_end] = forecast_segment
            labels.append(label)
            prov[forecast_start:forecast_end] = rec

        start = start + window
        end = min(end + window, N)
    return forecast, prov, labels

def convert_cputicks2mlcores(trace):
    return trace / (15 * 10 ** 6)

def plot_trace(trace, forecast=None, recommended_requests=None, plt_name="trace", y_label="CPU (milicores)", trace_legend="CPU Usage"):
    trace_len = len(trace)
    trace_idx = np.arange(trace_len) * 15
    if forecast is not None and recommended_requests is not None:
        trace_pd = pd.DataFrame({trace_legend: trace, "VPA recommended request": recommended_requests, "Forecast": forecast}, index=trace_idx)
    elif forecast is not None:
        trace_pd = pd.DataFrame({trace_legend: trace, "Forecast": forecast}, index=trace_idx)
    elif recommended_requests is not None:
        trace_pd = pd.DataFrame({trace_legend: trace, "VPA recommended request": recommended_requests}, index=trace_idx)
    else:
        trace_pd = pd.DataFrame({trace_legend: trace}, index=trace_idx)
    ax = trace_pd.plot()
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(y_label)
    ax.legend(loc='upper right')
    plt.savefig("./imgs/" + plt_name + ".pdf")
    plt.savefig("./imgs/" + plt_name + ".png")
    plt.show()

def main():
    global data_file, output_folder, config_file
    data = pickle.load(open(data_file, "rb"))

    if multiproc:
        pool_obj = multiprocessing.Pool(processes=None)
        results = pool_obj.starmap(get_all_recommendation, zip(data, np.repeat(recommender_config.TREE, len(data))))
        result_dict = {}
        for i in range(len(data)):
            result_dict[i] = {"forecast": results[i][0], "prov": results[i][1], "labels": results[i][2]}
        pickle.dump(result_dict, open(output_folder+ "/forecast_prov_{}".format(os.path.basename(data_file)), "wb"))
    else:
        result_dict = {}
        for i, trace in enumerate(data):
            logging.info("Processing trace {}".format(i))
            if "synthetic" not in data_file:
                trace = convert_cputicks2mlcores(trace[:600])
            forecast, prov, labels = get_all_recommendation(trace[:600], recommender_config.TREE, window=recommender_config.FORECASTING_WINDOW, sight=recommender_config.FORECASTING_SIGHT)
            result_dict[i] = {"forecast": forecast, "prov": prov, "labels": labels}
            plot_trace(trace[300:600], forecast[300:600], prov[300:600], plt_name="trace_{}".format(i), y_label="CPU (milicores)", trace_legend="CPU Usage")
        pickle.dump(result_dict, open(output_folder+ "/pando_forecast_prov_{}".format(os.path.basename(data_file)), "wb"))

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S', level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datafile', help="Data file")
    parser.add_argument('configfile', help="Tree file")
    parser.add_argument('outfolder', help="Output folder")
    parser.add_argument('-mp', "--multiproc", action='store_true', help='Compute recommendations in parallel')

    args = parser.parse_args()
    data_file = args.datafile
    config_file = args.configfile
    output_folder = args.outfolder
    multiproc = args.multiproc

    logging.info("Starting Pando VPA")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    main()






