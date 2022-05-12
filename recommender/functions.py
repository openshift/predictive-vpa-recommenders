import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import recommender_config
# import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning) #some modules such as ForecastingGridSearchCV when imported raise an annoying future warning

### Sktime native forecasters
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.compose import make_reduction

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.theta import ThetaForecaster

# pandas
import pandas as pd

### sklearn imports
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


### Statistical tests
import statsmodels.api as sm
# from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pymannkendall as mk
import ruptures as rpt


### Metrics
def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def rmse(y_pred,y_test):
    return np.sqrt(np.mean(np.power(y_pred - y_test,2)))

def smape(y_pred,y_test):
    nominator = np.abs(y_test - y_pred)
    denominator = np.abs(y_test) + np.abs(y_pred)
    return np.mean(2.0 * nominator / denominator) #the 2 in the nominator is because of symmetry

def mean_overprov(y_prov, y_test, margin=1.0, granularity=15):
    N = len(y_test)
    difference_over  = np.array(y_prov)  - np.array(y_test) * margin
    over_provision = difference_over > 0
    over_acc = sum(difference_over[over_provision]) / (granularity*N)
    return over_acc

def mean_underprov(y_prov, y_test, margin=1.0, granularity=15):
    N = len(y_test)
    difference_under = np.array(y_test) - np.array(y_prov) * margin
    under_provision = difference_under > 0
    under_acc = sum(difference_under[under_provision])  / (granularity*N)
    return under_acc

def perc_overprov(y_prov, y_test, margin=1.0, granularity=15):
    N = len(y_test)
    difference_over  = np.array(y_prov)  - np.array(y_test) * margin
    over_provision = difference_over > 0
    over_perc = (sum(over_provision == True)  / N)*100
    return over_perc

def perc_underprov(y_prov, y_test, margin=1.0, granularity=15):
    N = len(y_test)
    difference_under = np.array(y_test) - np.array(y_prov) * margin
    under_provision = difference_under > 0
    under_perc = (sum(under_provision == True) /N)*100
    return under_perc

### Functions to process time series
def convolution_filter(y, period):
    # Prepare Filter
    if period % 2 == 0:
        filt = np.array([.5] + [1] * (period - 1) + [.5]) / period
    else:
        filt = np.repeat(1. / period, period)

    # Signal Convolution
    conv_signal = signal.convolve(y, filt, mode='valid')

    # Padding (2-Sided Convolution)
    trim_head = int(np.ceil(len(filt) / 2.) - 1) or None
    trim_tail = int(np.ceil(len(filt) / 2.) - len(filt) % 2) or None

    if trim_head:
        conv_signal = np.r_[conv_signal, [np.nan] * trim_tail]
    if trim_tail:
        conv_signal = np.r_[[np.nan] * trim_head, conv_signal]

    return conv_signal


def compute_ses(y, alpha):
    nobs = len(y)  # X from the slides

    # Forecast Array
    fh = np.full(nobs + 1, np.nan)  # Initialize the Forecast array to NaNs # S from the slides
    fh[0] = y[0]  # Initialization of first value (instead of NaN)
    fh[1] = y[0]  # Initialization of first forecast

    # Simple Exponential Smoothing
    for t in range(2, nobs + 1):
        fh[t] = alpha * y[t - 1] + (1 - alpha) * fh[t - 1]  # s[t] = alpha * y....

    return (fh[:nobs], fh[nobs])


def forecast_ses(fh_next, start, end):
    ## Forecast Array
    fh_forecast = np.full(end - start, np.nan)
    fh_forecast[:] = fh_next

    return fh_forecast


def seasonal_decompose(y, period):
    nobs = len(y)

    # At least two observable periods in the trace
    if nobs < 2 * period:
        raise ValueError('lengh of signal must be larger than (2 * period)')

    # Convolution to retrieve step-by-step trend
    trend = convolution_filter(y, period)

    # Multiplicative de-trending to Retrieve average Season (period pattern)
    detrended = y / trend
    period_averages = np.array([np.nanmean(detrended[i::period], axis=0) for i in range(period)])
    period_averages /= np.mean(period_averages, axis=0)

    return period_averages  # "season" for deseasonalize


def deseasonalize(y, season):
    nobs = len(y)
    period = len(season)

    seasonal = np.tile(season, (nobs // period) + 1).T[:nobs]
    return y / seasonal


def reseasonalize(y, season, start):
    nobs = len(y)
    period = len(season)

    shift = period - (start % period)
    season = np.concatenate((season[-shift:], season[:-shift]))

    seasonal = np.tile(season, (nobs // period) + 1).T[:nobs]
    return y * seasonal


def compute_trend(y):
    lm = np.polyfit(np.arange(len(y)), y, 1)

    slope = lm[0]
    intercept = lm[1]
    drift = (slope * np.arange(0, len(y))) + intercept

    return (slope, intercept, drift)


def retrend(y, start, end, slope, intercept, type_t='mult'):
    if type_t == 'mult':
        #version 1
        drift = (slope * np.arange(start, end)) + intercept
        pred = y * (drift / np.mean(y))
    elif type_t =='add':
        #version 2 --> linear trend
        drift = slope * np.arange(start, end)
        pred = y + drift
    elif type_t=='no':
        pred = y
    return pred

def scan(trace, max_period = 100):

    errors = []
    test_size = len(trace) //8 # size of the test set for testing cycles
    if max_period is None: max_period = len(trace) //3 # max period to be examined

    y_train, y_test = trace[:-test_size],trace[-test_size:]

    true_max_period = min(max_period, int(len(y_train)/2))
    period_values = range(1, true_max_period)
    model = ThetaModel()

    for sp_val in period_values:
        model.fit(y_train, sp=sp_val)
        y_pred = model.forecast(test_size)[1]
        current_error = smape(y_pred, y_test)
        errors.append(current_error)

    period = period_values[np.argmin(errors)]
    return period, errors

## Functions to plot time series
def plot_series(series_list, labels, title='', legend=True):
    if len(series_list) != len(labels):
        raise ValueError('Number of series and labels must be the same')
    data = {}
    for i, series in enumerate(series_list):
        data[labels[i]] = series
    df = pd.DataFrame(data)
    df.plot(figsize=(12, 6))
    plt.title(title)
    if legend:
        plt.legend()
    plt.savefig("./tmp/" + title + ".png")
    plt.show()

### Own implementation of Theta Model
class ThetaModel:
    def __init__(self, alpha = 0.2):
        self.alpha = alpha


    def fit (self, y, sp):

        ## 1. Deseasonalize & Detrend
        season = seasonal_decompose(y, sp)            	                       ### THIS IS THE SEASON
        deseason = deseasonalize(y, season)                                      ### THIS IS THE DESEASONALIZED AND DETRENDED

        ## 2. Obtain Drift (general Trend) for later
        slope, intercept, drift = compute_trend(deseason)                              ### THIS IS THE SLOPE, INTERCEPT AND DRIFT

        ## 3. Obtain Simple Exponential Smoothing (SES)
        fitted, y_next = compute_ses(deseason, self.alpha)                                  ### THIS IS THE MODEL (Fitted, Next)

        ## Save "Model"
        self.season = season
        self.deseason = deseason
        self.slope = slope
        self.intercept = intercept
        self.drift = drift
        self.fitted = fitted
        self.next = y_next
        self.dataset = y
        self.last = len(y)

    def forecast (self, n_forecast):
        ## Get new boundaries
        start = self.last
        end = self.last + n_forecast
        ## 1. Forecast
        y_pred_1 = forecast_ses(self.next, start, end)

        ## 2. Re-Trend [FIRST]
        y_pred_2 = retrend(y_pred_1, start, end, self.slope, self.intercept, type_t='mult')

        ## 3. Re-Seasonalize [SECOND]
        y_pred = reseasonalize(y_pred_2, self.season, start)

        ## Join Full Trace
        full_trace_pred = np.concatenate((self.dataset, y_pred))

        return full_trace_pred, y_pred


### Forecasters with autotunning
def Theta_forecast(y, window, max_period=100, output=False, title=''):
    sp, _ = scan(y, max_period)
    forecaster = ThetaModel()
    forecaster.fit(y, sp)
    _, y_pred = forecaster.forecast(window)
    if output: plot_series([pd.Series(y), pd.Series(y_pred, index=np.arange(len(y), len(y) + window))],
                           labels=["y", "y_pred"], title=title)
    if output: print("Detected period: {}".format(sp))
    return y_pred

def Theta_forecast_sktime(y, window, max_period = 100, output =False, title=''):
    y = pd.Series(y)
    fh = np.arange(1, window+1)
    forecaster = ThetaForecaster()
    sps = np.arange(1,max_period)
    param_grid = {"sp": sps}
    cv = SlidingWindowSplitter(window_length=int(len(y) * recommender_config.WINDOW_SPILLTER), fh = fh)
    gscv = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)
    gscv.fit(y)
    y_pred = gscv.best_forecaster_.predict(fh)
    if output: plot_series([y, y_pred], labels=["y","y_pred"], title=title)
    return y_pred.values

def Naive_forecast(y, window, max_period = 100, output =False, title=''):
    y = pd.Series(y)
    fh = np.arange(1, window+1)
    forecaster = NaiveForecaster()
    sps = np.arange(1,max_period)
    param_grid = {"strategy" : ["last", "drift"], "sp": sps}
    cv = SlidingWindowSplitter(window_length=int(len(y) * recommender_config.WINDOW_SPILLTER), fh = fh)
    gscv = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)
    gscv.fit(y)
    y_pred = gscv.predict(fh)
    if output: plot_series([y, y_pred], labels=["y","y_pred"], title=title)
    return y_pred.values

def lr_forecast(y, window, output =False, title=''):
    y = pd.Series(y)
    fh = np.arange(1, window+1)
    forecaster = PolynomialTrendForecaster()
    param_grid = {"degree" : [1,2]}
    cv = SlidingWindowSplitter(window_length=int(len(y) * recommender_config.WINDOW_SPILLTER), fh = fh)
    gscv = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)
    gscv.fit(y, fh=fh)
    y_pred = gscv.predict(fh)
    if output: plot_series([y, y_pred], labels=["y","y_pred"], title=title)
    return y_pred.values

def autoarima_forecast(y, window, output =False, title=''):
    y = pd.Series(y)
    fh = np.arange(1, window+1)
    forecaster = AutoARIMA()
    forecaster.fit(y, fh=fh)
    y_pred = forecaster.predict(fh)
    if output: plot_series([y, y_pred], labels=["y","y_pred"], title=title)
    return y_pred.values

def KNN_forecast(y, window, output =False, title=''):
    y = pd.Series(y)
    fh = np.arange(1, window+1)
    regressor = KNeighborsRegressor()
    forecaster = make_reduction(regressor, strategy="recursive")
    param_grid = {"window_length": [7, 12, 15], "estimator__n_neighbors": np.arange(1, 5)}
    cv = SlidingWindowSplitter(window_length=int(len(y) * recommender_config.WINDOW_SPILLTER), fh = fh)
    gscv = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)
    gscv.fit(y)
    y_pred = gscv.predict(fh)
    if output: plot_series([y, y_pred], labels=["y", "y_pred"], title=title)
    return y_pred.values

def SVM_forecast(y, window, output =False, title=''):
    y = pd.Series(y)
    fh = np.arange(1, window+1)
    regressor = SVR()
    forecaster = make_reduction(regressor, strategy="recursive")
    param_grid = {"window_length": [7, 12, 15],
                  "estimator__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                  "estimator__gamma": [1e-3, 1e-4],
                  "estimator__C": [1, 10, 100, 1000]}
    cv = SlidingWindowSplitter(window_length=int(len(y) * recommender_config.WINDOW_SPILLTER), fh = fh)
    gscv = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)
    gscv.fit(y)
    y_pred = gscv.predict(fh)
    if output: plot_series([y, y_pred], labels=["y", "y_pred"], title=title)
    return y_pred.values

def DT_forecast(y, window, output =False, title=''):
    y = pd.Series(y)
    fh = np.arange(1, window+1)
    regressor = DecisionTreeRegressor()
    forecaster = make_reduction(regressor, strategy="recursive")
    param_grid = {"window_length": [7, 12, 15]}
    cv = SlidingWindowSplitter(window_length=int(len(y) * recommender_config.WINDOW_SPILLTER), fh = fh)
    gscv = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)
    gscv.fit(y)
    y_pred = gscv.predict(fh)
    if output: plot_series([y, y_pred], labels=["y", "y_pred"], title=title)
    return y_pred.values

def VPA_forecast(y, window, output =False, title=''):
    pre_step_idxs = range(len(y)-window, len(y))
    previous_usage = y[pre_step_idxs]

    y = pd.Series(y)
    y_pred = previous_usage
    y_pred_index = range(len(y), len(y)+window)
    y_pred = pd.Series(y_pred, index=y_pred_index)

    if output: plot_series([y, y_pred], labels=["y", "y_pred"], title=title)

    return y_pred.values

### Statistical tests
def detect_theta_periodicity(trace, theta_threshold):
    period_est, error_est = scan(trace)
    return np.var(error_est) >= theta_threshold

def detect_adf_stationarity(trace, stat_threshold):
    adf = sm.tsa.stattools.adfuller(trace)
    stationary = adf[1] < stat_threshold # if True reject null hypothesis --> stationary
    return stationary

def detect_trend(trace):
    return mk.original_test(trace)[0] != 'no trend'

def detect_change_point(trace, max_change_points):
    algo1 = rpt.Pelt(model="rbf").fit(trace)
    change_location = algo1.predict(pen=max_change_points)
    return len(change_location)>1

def perform_tests(y, stat_threshold= 0.01, theta_threshold = 0.001, max_change_points = 5):
    theta_test = detect_theta_periodicity(y, theta_threshold)
    adf_test = detect_adf_stationarity(y, stat_threshold)
    mk_test = detect_trend(y)
    cp_test = detect_change_point(y, max_change_points)
    return {"theta":theta_test, "adf":adf_test, "mk": mk_test, "pelt":cp_test}