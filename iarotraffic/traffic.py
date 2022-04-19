# Import dependencies

from typing import Sequence
from numpy.core.fromnumeric import size
from numpy.core.numeric import _moveaxis_dispatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pandas.core.algorithms import value_counts
import requests
import datetime
from scipy.stats.mstats import hmean
import os
import pathlib
from pystoned import CQER, wCQER, CQERG
from pystoned.constant import CET_ADDI, FUN_PROD, OPT_LOCAL, RTS_VRS
from pystoned import dataset as dataset
from pystoned.plot import plot2d, plot2d_test
import pyarrow
import sys
import math
import scipy

DEF_COL_NAMES = ['id', 'year', 'day', 'hour',
                 'minute', 'second', 'hund_second', 'length', 'lane',
                 'direction', 'vehicle', 'speed', 'faulty', 'total_time',
                 'time_interval', 'queue_start']
DEF_FILEPATH = ''
DEF_AGG_TIME_PER = 5


def download_lam_day_report(tms_id: str, region: str, year: int, day: int,
                            time_from: int = 6, time_to: int = 20, if_faulty: bool = True) -> pd.DataFrame:
    """Download the lam-report for the specified day. Downloaded data is cleaned.

    Parameters
    ----------
    tms_id : str
        The id number of selected traffic measurement station (TMS). \
        Meta-data about TMS is available: https://www.digitraffic.fi/en/road-traffic/#current-data-from-tms-stations .\
        String format is used, as `tms_id` might have a leading zero.
    region : str
        The code of the region, where the selected traffic measurement station is located.
        For values check "Accessing the data" article by the link: \
        https://vayla.fi/en/transport-network/data/open-data/road-network/tms-data#accessing-the-data.\
        String format is used, as `region` might have a leading zero.
    year : int
        The year data was collected in the 4-digit format.
    day : int
        The day of the year the data was collected. \
        The day is provide as an integer in range(1, 366), with 1 - January 1st, \
        365 (366 in leap year) - December 31st. \
        To caclulate the day of the year you can use `iarotraffic.date_to_day()` function.
    time_from : int, optional
        Data is cleaned, and the `time_from` shows, from which hour of the day the data is kept in 24h format. \
        (The default is 6, which reffrs to 6am)
    time_to : int, optional
        Data is cleaned, and the `time_to` shows, till which hour of the day the data is kept in 24h format. \
        (The default is 20, which refers to 8pm)
    if_faulty : Bool, optional
        The default is `True`, meaning that faulty data is deleted. \
        Faulty data is detected based on information from the dataset.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame containing cleaned lam-report for the selected TMS station \
        for the selected day of the selected year.

    See Also
    --------
    iarotraffic.date_to_day :
        The function for conversion the date to the day of the year.

    """
    start_time = time.perf_counter()
    column_names = DEF_COL_NAMES
    df = pd.DataFrame()
    url = 'https://tie-test.digitraffic.fi/api/tms/history/raw/lamraw_TMS_YY_DD.csv'

    # Create the actual url
    url = url.replace('YYYY', str(year)).replace('REGION_ID', region).replace(
        'TMS', tms_id).replace('YY', str(year)[2:4]).replace('DD', str(day))

    # Try to download the file
    if requests.get(url).status_code != 404:

        # Downloading the file from the server
        df = pd.read_csv(url, delimiter=";", names=column_names)

        # Assigning dates
        df['date'] = datetime.date(year, 1, 1) + datetime.timedelta(day - 1)
        df['cars'] = df['vehicle'].apply(lambda x: 1 if x == 1 else 0)
        df['buses'] = df['vehicle'].apply(lambda x: 1 if x == 3 else 0)
        df['trucks'] = df['vehicle'].apply(lambda x: 1 if x == 2 or x == 4 or x == 5 or x == 6 or x == 7 else 0)

        # Deleting faulty data point.
        if if_faulty is True:
            df = df[df.faulty != 1]

        # Selecting data only from the specified timeframe
        # df['total_time'] = pd.to_numeric(df['total_time'])
        df = df[df.total_time >= time_from*60*60*100]
        df = df[df.total_time <= time_to*60*60*100]

        end_time = time.perf_counter()
        print(f"Download successful - file for the sensor {tms_id} for the day {day} in year {year}. \
                Download took {end_time-start_time:0.4f} seconds")
    else:
        print(
            f"File for the sensor {tms_id} for the day {day} in year {year} doesn't exist. ")
        return df

    return df

def date_to_day(date_to_change: datetime.date) -> int:
    first_day = datetime.date(date_to_change.year, 1, 1)
    day = abs(date_to_change - datetime.date(date_to_change.year, 1, 1)).days+1
    return day

def day_to_date(year: int, day: int) -> datetime.date:
    date = datetime.date(year, 1, 1) + datetime.timedelta(days=day-1)
    return date

def previous_days(year: int, day: int, num_of_days: int, lam_id, region):
    prediction_date = day_to_date(year, day)
    days_list = [None] * num_of_days
    for i in range(num_of_days):
        days_list[i] = [lam_id, region, year-i-1, day+5-day_to_date(year-i-1, day).isoweekday()]
    return days_list


"""LOADING TRAFFIC DATA FROM LOCAL GZIP-FILE.
IF UNSUCCESSFUL, LOADING DATA FROM THE SERVER AND SAVING IT LOCALLY AS GZIP-FILE"""

def traffic_data_load(
        tms_id: str, region: str, year: int, day_from: int, day_to: int, time_from=6, time_to=20,
        input_path=None, input_name=None, file_type='gzip') -> pd.DataFrame:
    start_time = time.perf_counter()
    filename = 'data' + '_' + tms_id + '_' + str(year)[2:4] \
        + '_' + str(day_from) + '_' + str(day_to) + '_' + str(time_from) \
        + 'h_' + str(time_to) + 'h.gzip'
    filepath = DEF_FILEPATH
    column_names = DEF_COL_NAMES
    df = pd.DataFrame()

    # Alternative name selection
    if input_name is not None:
        filename = input_name

    # Alternative name selection
    if input_path is not None:
        filepath = input_path

    # Make a unified location identifier
    filepath = pathlib.Path(filepath)
    path = filepath / filename

    # Use the gzip file
    if file_type == 'gzip':

        # First trying to load file locally
        if (os.path.exists(path) is True) and (os.path.getsize(path) != 0):
            df = pd.read_parquet(path)
        else:
            print(
                f"File {path} doesn't exist. Trying to download data from the server and save locally...")

            # Second trying to download files from online
            start_time_gzip = time.perf_counter()
            for day in range(day_from, day_to + 1):
                if df.empty:
                    df = download_lam_day_report(
                        tms_id, region, year, day, time_from=time_from, time_to=time_to)
                else:
                    df = df.append(download_lam_day_report(
                        tms_id, region, year, day, time_from=time_from, time_to=time_to), ignore_index=True)
            end_time_gzip = time.perf_counter()
            print(
                f"Loading file from server took {end_time_gzip-start_time_gzip:0.4f} seconds. Saving .gzip file...")

            # Saving the .gzip file with downloaded data
            start_time_gzip = time.perf_counter()
            df.to_parquet(path, engine='pyarrow', compression='gzip')
            end_time_gzip = time.perf_counter()
            print(
                f"Saving .gzip file took {end_time_gzip-start_time_gzip:0.4f} seconds")

    end_time = time.perf_counter()
    if df.empty:
        print(f"Loading unsuccessfull. Check the parameters and avaiability of data.")
    else:
        print(
            f"Loading successfull. Data loading took {end_time-start_time:0.4f} seconds")

    # The Pandas DataFrame is returned, containing the data for the selected period
    return df

def traffic_data_load_from_list(
        days_list, time_from=6, time_to=20,
        input_path=None, input_name=None, file_type='gzip') -> pd.DataFrame:
    """
    list in a format [[lam_id, region, year, day]]
    """
    start_time = time.perf_counter()
    filename = 'data' + '_' + days_list[0][0] + '_' + str(days_list[0][2])[2:4] \
        + '_' + str(days_list[0][3]) + '_' + str(len(days_list)) + '_' + str(time_from) \
        + 'h_' + str(time_to) + 'h.gzip'
    filepath = DEF_FILEPATH
    column_names = DEF_COL_NAMES
    df = pd.DataFrame()

    # Alternative name selection
    if input_name is not None:
        filename = input_name

    # Alternative name selection
    if input_path is not None:
        filepath = input_path

    # Make a unified location identifier
    filepath = pathlib.Path(filepath)
    path = filepath / filename

    # Use the gzip file
    if file_type == 'gzip':

        # First trying to load file locally
        if (os.path.exists(path) is True) and (os.path.getsize(path) != 0):
            df = pd.read_parquet(path)
        else:
            print(
                f"File {path} doesn't exist. Trying to download data from the server and save locally...")

            # Second trying to download files from online
            start_time_gzip = time.perf_counter()
            for count, value in enumerate(days_list):
                if df.empty:
                    df = download_lam_day_report(
                        value[0], value[1], value[2], value[3], time_from=time_from, time_to=time_to)
                else:
                    df = df.append(download_lam_day_report(
                        value[0], value[1], value[2], value[3], time_from=time_from, time_to=time_to),
                        ignore_index=True)
            end_time_gzip = time.perf_counter()
            print(
                f"Loading file from server took {end_time_gzip-start_time_gzip:0.4f} seconds. Saving .gzip file...")

            # Saving the .gzip file with downloaded data
            start_time_gzip = time.perf_counter()
            df.to_parquet(path, engine='pyarrow', compression='gzip')
            end_time_gzip = time.perf_counter()
            print(
                f"Saving .gzip file took {end_time_gzip-start_time_gzip:0.4f} seconds")

    end_time = time.perf_counter()
    if df.empty:
        print(f"Loading unsuccessfull. Check the parameters and avaiability of data.")
    else:
        print(
            f"Loading successfull. Data loading took {end_time-start_time:0.4f} seconds")

    # The Pandas DataFrame is returned, containing the data for the selected period
    return df


""" PROCESSING OF THE TRAFFIC DATA FRAME: CALCULATION OF SPACE-MEAN SPEED AND SPACE-MEAN FLOW.
BASED ON THAT THE DENSITY IS CALCULATED """


def flow_speed_calculation(df: pd.DataFrame, aggregation_time_period=DEF_AGG_TIME_PER) -> pd.DataFrame:
    start_time = time.perf_counter()
    time_agg = pd.DataFrame()
    space_agg = pd.DataFrame()

    # Create the aggregation parametere based on aggregation_time_period
    df['aggregation'] = (df.hour * 60 + df.minute)/aggregation_time_period
    df = df.astype({'aggregation': int})

    # Aggregate flow and speed by time
    time_agg = df.groupby(['id', 'date', 'aggregation', 'direction', 'lane'],
                          as_index=False).agg(time_mean_speed=('speed', 'mean'),
                                              flow=('speed', 'count'),
                                              cars=('cars', 'sum'),
                                              buses=('buses', 'sum'),
                                              trucks=('trucks', 'sum'))
    time_agg['hourlyflow'] = 60/aggregation_time_period * time_agg.flow
    time_agg['hourlycars'] = 60/aggregation_time_period * time_agg.cars
    time_agg['hourlybuses'] = 60/aggregation_time_period * time_agg.buses
    time_agg['hourlytrucks'] = 60/aggregation_time_period * time_agg.trucks
    time_agg['qdivv'] = time_agg['hourlyflow'].div(
        time_agg['time_mean_speed'].values)

    # Aggregate flow and speed by space and calculate density. Calculate the weights
    space_agg = time_agg.groupby(['id', 'date', 'aggregation', 'direction'],
                                 as_index=False).agg(qdivvsum=('qdivv', 'sum'),
                                                     flow=('hourlyflow', 'sum'),
                                                     cars=('hourlycars', 'sum'),
                                                     buses=('hourlybuses', 'sum'),
                                                     trucks=('hourlytrucks', 'sum'))
    space_agg['space_mean_speed'] = 1/(space_agg.qdivvsum.div(space_agg.flow))
    space_agg['density'] = space_agg.flow.div(space_agg.space_mean_speed)
    space_agg['weight'] = float(1/len(space_agg))
    space_agg['car_proportion'] = space_agg.cars.div(space_agg.flow)
    space_agg['bus_proportion'] = space_agg.buses.div(space_agg.flow)
    space_agg['truck_proportion'] = space_agg.trucks.div(space_agg.flow)

    end_time = time.perf_counter()
    print(
        f"Aggregating data for modeling took {end_time-start_time:0.4f} seconds")

    return space_agg


""" BAGGING OF PROCESSED DATA """
def bagging(dirdata: pd.DataFrame, grid_size_x=70, grid_size_y=400) -> pd.DataFrame:
    bagged_data = pd.DataFrame()

    # Getting the max density and flow values to calculcate the size of the bag
    maxDensity = dirdata.density.max()
    maxFlow = dirdata.hourlyflow.max()

    # Calclulating the size of the bag
    grid_density_size = maxDensity / grid_size_x
    grid_flow_size = maxFlow / grid_size_y

    # Assigning the bag number for density and
    dirdata['grid_density'] = dirdata.density / grid_density_size
    dirdata['grid_flow'] = dirdata.hourlyflow / grid_flow_size
    dirdata = dirdata.astype({'grid_density': int, 'grid_flow': int})

    # Calculating the centroid and the weight of each bag
    bagged_data = dirdata.groupby(
        ['id', 'direction', 'grid_density', 'grid_flow'],
        as_index=False).agg(
            bag_size=('id', 'count'), sum_flow=('hourlyflow', 'sum'), sum_density=('density', 'sum'))
    bagged_data['centroid_flow'] = bagged_data.sum_flow.div(bagged_data.bag_size)
    bagged_data['centroid_density'] = bagged_data.sum_density.div(bagged_data.bag_size)
    bagged_data['weight'] = bagged_data.bag_size.div(len(dirdata))

    return bagged_data, maxDensity, maxFlow, dirdata

def representor(alpha: Sequence[float], beta: Sequence[float], x: float) -> float:
    """
    Calculation of representation function (Kuosmanen, 2008 / Formula 4.1)
    g_hat_min = min{alpha_i_hat + beta_i_hat * x}

    alpha: np.array of alphas
    beta: np.array of betas
    x: float x

    returns the minimum value g_hat for the given x
    """
    g_hat = np.empty_like(alpha)
    x_arr = np.full_like(alpha, x, dtype=float)
    g_hat = alpha + beta * x_arr
    g_hat_min = np.amin(g_hat)

    return g_hat_min

def out_of_sample_mse(model, test_array: np.ndarray) -> list:
    """
    bm_array order: 0 - x_train, 1 - y_train, 2 - beta, 3 - alpha, 4 - residual, 5 - y_train_calc,
        6 - y_train_calc-y_act, 7 - residual squared, 8 - abs(y_train_calc-y_act)
    test_array order: 0 - x_test, 1 - y_test, 2 - representor, 3 - residual, 4 - residual squared, 5 - abs(residual)
    """
    bm_array = np.column_stack((model.x, model.y))
    bm_array.view('f8,f8').sort(order=['f0'], axis=0)
    flatten = model.get_beta().flatten()
    bm_array = np.column_stack(
        (bm_array, flatten))
    bm_array = np.column_stack(
        (bm_array, model.get_alpha()))
    bm_array = np.column_stack(
        (bm_array, model.get_residual()))
    bm_array = np.column_stack(
        (bm_array, bm_array[:, 0] * bm_array[:, 2] + bm_array[:, 3]))
    bm_array = np.column_stack(
        (bm_array, bm_array[:, 1] - bm_array[:, 5]))
    bm_array = np.column_stack(
        (bm_array, np.square(bm_array[:, 4])))
    bm_array = np.column_stack(
        (bm_array, abs(bm_array[:, 1] - bm_array[:, 5])))

    # Sorting of test array
    test_array.view('f8,f8').sort(order=['f0'], axis=0)

    # Calculation of y_test_hat using representor function
    test_array = np.append(test_array, np.zeros((len(test_array), 1), dtype=float), axis=1)

    for i in range(len(test_array[:, 0])):
        test_array[i, 2] = representor(bm_array[:, 3], bm_array[:, 2], test_array[i, 0])

    test_array = np.column_stack(
        (test_array, test_array[:, 1] - test_array[:, 2]))

    test_array = np.column_stack(
        (test_array, np.square(test_array[:, 3])))

    test_array = np.column_stack(
        (test_array, abs(test_array[:, 1] - test_array[:, 2])))

    # with np.printoptions(threshold=np.inf):
    #    print(test_array)

    train_mse = np.sum(bm_array[:, 7]) / len(bm_array[:, 7])
    print(f"Train set MSE equals {train_mse}")

    train_rmse = math.sqrt(np.sum(bm_array[:, 7]) / len(bm_array[:, 7]))
    print(f"Train set RMSE equals {train_rmse}")

    train_mae = np.sum(bm_array[:, 8]) / len(bm_array[:, 7])
    print(f"Train set MAE equals {train_mae}")
    print('\n')

    test_mse = np.sum(test_array[:, 4]) / len(test_array[:, 4])
    print(f"Test MSE equals {test_mse}")

    test_rmse = math.sqrt(np.sum(test_array[:, 4]) / len(test_array[:, 4]))
    print(f"Test RMSE equals {test_rmse}")

    test_mae = np.sum(test_array[:, 5]) / len(test_array[:, 5])
    print(f"Test MAE equals {test_mae}")

    error_list = [[train_mse, train_rmse, train_mae], [test_mse, test_rmse, test_mae]]

    print("Calculation of out of sample MSE completed")

    return error_list

def in_sample_mse(model) -> list:
    """
    bm_array order: 0 - x_train, 1 - y_train, 2 - beta, 3 - alpha, 4 - residual, 5 - y_train_calc,
        6 - y_train_calc-y_act, 7 - residual squared, 8 - abs(y_train_calc-y_act)
    """
    bm_array = np.column_stack((model.x, model.y))
    bm_array.view('f8,f8').sort(order=['f0'], axis=0)
    flatten = model.get_beta().flatten()
    bm_array = np.column_stack(
        (bm_array, flatten))
    bm_array = np.column_stack(
        (bm_array, model.get_alpha()))
    bm_array = np.column_stack(
        (bm_array, model.get_residual()))
    bm_array = np.column_stack(
        (bm_array, bm_array[:, 0] * bm_array[:, 2] + bm_array[:, 3]))
    bm_array = np.column_stack(
        (bm_array, bm_array[:, 1] - bm_array[:, 5]))
    bm_array = np.column_stack(
        (bm_array, np.square(bm_array[:, 4])))
    bm_array = np.column_stack(
        (bm_array, abs(bm_array[:, 1] - bm_array[:, 5])))

    train_mse = np.sum(bm_array[:, 7]) / len(bm_array[:, 7])
    print(f"Train set MSE equals {train_mse}")

    train_rmse = math.sqrt(np.sum(bm_array[:, 7]) / len(bm_array[:, 7]))
    print(f"Train set RMSE equals {train_rmse}")

    train_mae = np.sum(bm_array[:, 8]) / len(bm_array[:, 7])
    print(f"Train set MAE equals {train_mae}")
    print('\n')

    error_list = [train_mse, train_rmse, train_mae]

    print("Calculation of out of sample MSE completed")

    return error_list


def iaroplot_diff_models(
        x_original,
        y_original,
        proportion,
        x_bagged,
        y_bagged,
        weight,
        bagged_model_x, bagged_model_y, bagged_model_f, orig_model_x, orig_model_y, orig_model_f, suptitle):

    plt.clf()
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # [0; 0]
    axs[0, 0].scatter(
        x_original,
        y_original,
        marker='.',
        c="k")
    axs[0, 0].set_title("original data scatter")

    # [0; 1]
    axs[0, 1].scatter(
        x_original,
        y_original,
        c=proportion,
        marker='.',
        cmap="RdYlGn")
    axs[0, 1].set_title("car proportion data scatter")

    # [0; 2]
    axs[0, 2].scatter(
        x_bagged,
        y_bagged,
        c='k',
        marker='o',
        s=weight*10000)
    axs[0, 2].set_title("bagged data scatter")

    # [1; 0]
    x = np.array(bagged_model_x).T[0]
    y = np.array(bagged_model_y).T
    yhat = np.array(bagged_model_f).T
    data = (np.stack([x, y, yhat], axis=0)).T
    # sort
    data = data[np.argsort(data[:, 0])].T
    x, y, f = data[0], data[1], data[2]

    axs[1, 0].scatter(x, y, color="k", marker='x')
    axs[1, 0].plot(x, f, color="r")
    axs[1, 0].set_title("wCQR")

    # [1; 1]
    x_orig = np.array(orig_model_x).T[0]
    y_orig = np.array(orig_model_y).T
    yhat_orig = np.array(orig_model_f).T
    data_orig = (np.stack([x_orig, y_orig, yhat_orig], axis=0)).T
    # sort
    data_orig = data_orig[np.argsort(data_orig[:, 0])].T
    x_orig, y_orig = data_orig[0], data_orig[1]
    axs[1, 1].scatter(x_orig, y_orig, color="k", marker='x')
    axs[1, 1].plot(x, f, color="r")
    axs[1, 1].set_title("wCQR on original data")

    # [1; 2]
    x = np.array(orig_model_x).T[0]
    y = np.array(orig_model_y).T
    yhat = np.array(orig_model_f).T
    data = (np.stack([x, y, yhat], axis=0)).T
    # sort
    data = data[np.argsort(data[:, 0])].T
    x, y, f = data[0], data[1], data[2]
    axs[1, 2].scatter(x, y, color="k", marker='x')
    axs[1, 2].plot(x, f, color="r")
    axs[1, 2].set_title("CNLSG")

    fig.suptitle(suptitle, fontsize=16)

    plt.savefig("Fig")
    plt.close()

    return None

def iaroplot_days(
        x_test,
        y_test,
        w_test,
        x_train,
        y_train,
        w_train,
        train_model_x, train_model_y, train_model_f, test_model_x, test_model_y, test_model_f, test_data, suptitle):

    plt.clf()
    fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex='all', sharey='all')

    # [0; 0]
    axs[0, 0].scatter(
        x_train,
        y_train,
        c='k',
        marker='.',
        s=w_train*10000,
        label="TRAIN data")
    axs[0, 0].set_title("TRAIN data scatter")
    axs[0, 0].legend()

    # [0; 1]
    """
    axs[0, 1].scatter(
        x_original,
        y_original,
        c=proportion,
        marker='.',
        cmap="RdYlGn")
    axs[0, 1].set_title("car proportion data scatter")
    """

    # [0; 1]
    axs[0, 1].scatter(
        x_test,
        y_test,
        c='k',
        marker='.',
        s=w_test*10000,
        label="TEST data")
    axs[0, 1].set_title("TEST data scatter")
    axs[0, 1].legend()

    # [1; 0]
    x = np.array(train_model_x).T[0]
    y = np.array(train_model_y).T
    yhat = np.array(train_model_f).T
    data = (np.stack([x, y, yhat], axis=0)).T
    # sort
    data = data[np.argsort(data[:, 0])].T
    x, y, f = data[0], data[1], data[2]

    axs[1, 0].scatter(x, y, color="k", marker='x', label="TRAIN data")
    axs[1, 0].plot(x, f, color="g", label="TRAIN model")
    axs[1, 0].set_title("TRAIN model")
    axs[1, 0].legend()

    # [1; 1]
    x = np.array(test_model_x).T[0]
    y = np.array(test_model_y).T
    yhat = np.array(test_model_f).T
    data = (np.stack([x, y, yhat], axis=0)).T
    # sort
    data = data[np.argsort(data[:, 0])].T
    x, y, f = data[0], data[1], data[2]
    axs[1, 1].scatter(x, y, color="k", marker='x', label="TEST data")
    axs[1, 1].plot(x, f, color="r", label="TEST model")
    axs[1, 1].plot(test_data[:, 0], test_data[:, 2], color="g", label="TRAIN model")
    axs[1, 1].set_title("TEST and TRAIN model")
    axs[1, 1].legend()

    fig.suptitle(suptitle, fontsize=16)

    plt.savefig("Fig")

    return None

def compare_models(bagged_data: pd.DataFrame,
                   original_data: pd.DataFrame, month: str, year: int, tau: float = 0.5, select_direction: int = 2):

    start_time = time.perf_counter()
    x_bag = bagged_data[bagged_data.direction == select_direction].centroid_density
    y_bag = bagged_data[bagged_data.direction == select_direction].centroid_flow
    w_bag = bagged_data[bagged_data.direction == select_direction].weight

    x_orig = np.array(original_data[original_data.direction == select_direction].density).reshape(
        len(original_data[original_data.direction == select_direction]), 1)
    y_orig = np.array(original_data[original_data.direction == select_direction].flow).reshape(
        len(original_data[original_data.direction == select_direction]), 1)

    test_array = np.column_stack((x_orig, y_orig))

    """
    fig_name = "Bagged data scatter - " + month + "_" + str(year) + "_tau_" + str(int(100*tau))
    plt.scatter(
        bagged_data[bagged_data.direction == select_direction].centroid_density,
        bagged_data[bagged_data.direction == select_direction].centroid_flow,
        c='b',
        marker='.',
        s=bagged_data[bagged_data.direction == select_direction].weight*10000,
        label="Bagged data scatter")
    plt.savefig(fname=fig_name)
    plt.clf()"""

    """
    fig_name = "Original data scatter - " + month + "_" + str(year) + "_tau_" + str(int(100*tau))
    plt.scatter(
        original_data[original_data.direction == select_direction].density,
        original_data[original_data.direction == select_direction].flow,
        c=original_data[original_data.direction == select_direction].car_proportion,
        marker='.',
        cmap="RdYlGn",
        label="Original data scatter")
    plt.savefig(fname=fig_name)
    plt.clf()"""

    bagged_model = wCQER.wCQR(y=y_bag, x=x_bag, w=w_bag, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
    bagged_model.optimize(OPT_LOCAL)

    original_model = CQERG.CQRG(y_orig, x_orig, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
    original_model.optimize(OPT_LOCAL)

    fig_name = "Bagged model on bagged data - " + month + "_" + str(year) + "_tau_" + str(int(100*tau))
    plot2d(
        bagged_model, x_select=0,
        label_name="Bagged model on bagged data", fig_name=fig_name)
    plt.clf()

    """
    fig_name = "Bagged model on original data - " + month + "_" + str(year) + "_tau_" + str(int(100*tau))
    plot2d_test(
        bagged_model, test_array, x_select=0,
        label_name="Bagged model on original data", fig_name=fig_name)
    plt.clf()"""

    """
    fig_name = "Original model on original data - " + month + "_" + str(year) + "_tau_" + str(int(100*tau))
    plot2d(
        original_model, x_select=0,
        label_name="Original model on original data", fig_name=fig_name)
    plt.clf()
    """
    """
    bm_array order:
    0 - x_train, 1 - y_train, 2 - beta, 3 - alpha, 4 - residual, 5 - residual squared,
    6 - abs(residual)
    """

    bm_array = np.column_stack((bagged_model.x, bagged_model.y))
    bm_array.view('f8,f8').sort(order=['f0'], axis=0)
    flatten = bagged_model.get_beta().flatten()
    bm_array = np.column_stack(
        (bm_array, flatten))
    bm_array = np.column_stack(
        (bm_array, bagged_model.get_alpha()))
    bm_array = np.column_stack(
        (bm_array, bagged_model.get_residual()))
    bm_array = np.column_stack(
        (bm_array, np.square(bm_array[:, 4])))
    bm_array = np.column_stack(
        (bm_array, abs(bm_array[:, 4])))

    """
    orig_array order:
    0 - x_train, 1 - y_train, 2 - beta, 3 - alpha, 4 - residual, 5 - residual squared,
    6 - abs(residual)
    """

    orig_array = np.column_stack((original_model.x, original_model.y))
    orig_array.view('f8,f8').sort(order=['f0'], axis=0)
    flatten_orig = original_model.get_beta().flatten()
    orig_array = np.column_stack(
        (orig_array, flatten_orig))
    orig_array = np.column_stack(
        (orig_array, original_model.get_alpha()))
    orig_array = np.column_stack(
        (orig_array, original_model.get_residual()))
    orig_array = np.column_stack(
        (orig_array, np.square(orig_array[:, 4])))
    orig_array = np.column_stack(
        (orig_array, abs(orig_array[:, 4])))

    """
    test_array order: 0 - x_test, 1 - y_test, 2 - bagged_representor, 3 - bagged_residual,
    4 - bagged_residual squared, 5 - abs(residual),
    6 - orig_representor, 7 - residual btw orig and bm, 8 - 7 squared, 9 - abs(7)
    """
    test_array.view('f8,f8').sort(order=['f0'], axis=0)

    # Calculation of y_test_hat using representor function
    test_array = np.append(test_array, np.zeros((len(test_array), 1), dtype=float), axis=1)

    for i in range(len(test_array[:, 0])):
        test_array[i, 2] = representor(bm_array[:, 3], bm_array[:, 2], test_array[i, 0])

    test_array = np.column_stack(
        (test_array, test_array[:, 1] - test_array[:, 2]))

    test_array = np.column_stack(
        (test_array, np.square(test_array[:, 3])))

    test_array = np.column_stack(
        (test_array, abs(test_array[:, 3])))

    test_array = np.append(test_array, np.zeros((len(test_array), 1), dtype=float), axis=1)

    for i in range(len(test_array[:, 0])):
        test_array[i, 6] = representor(orig_array[:, 3], orig_array[:, 2], test_array[i, 0])

    test_array = np.column_stack(
        (test_array, test_array[:, 6] - test_array[:, 2]))

    test_array = np.column_stack(
        (test_array, np.square(test_array[:, 7])))

    test_array = np.column_stack(
        (test_array, abs(test_array[:, 7])))

    max_density_bagged = test_array[np.argmax(test_array, axis=0)[2]][0]
    max_flow_bagged = np.max(test_array[:, 2])

    max_density_orig = test_array[np.argmax(test_array, axis=0)[6]][0]
    max_flow_orig = np.max(test_array[:, 6])

    print(f"Max density of bagged model is {max_density_bagged}")
    print(f"Max flow of bagged model is {max_flow_bagged}")

    print(f"Max density of original model is {max_density_orig}")
    print(f"Max flow of original model is {max_flow_orig}")

    bm_mse = np.sum(bm_array[:, 5]) / len(bm_array[:, 5])
    print(f"Bagged model on bagged data MSE equals {bm_mse}")

    bm_rmse = math.sqrt(np.sum(bm_array[:, 5]) / len(bm_array[:, 5]))
    print(f"Bagged model on bagged data RMSE equals {bm_rmse}")

    bm_mae = np.sum(bm_array[:, 6]) / len(bm_array[:, 6])
    print(f"Bagged model on bagged data MAE equals {bm_mae}")
    print('\n')

    bm_orig_mse = np.sum(test_array[:, 4]) / len(test_array[:, 4])
    print(f"Bagged model on original data MSE equals {bm_orig_mse}")

    bm_orig_rmse = math.sqrt(np.sum(test_array[:, 4]) / len(test_array[:, 4]))
    print(f"Bagged model on original data RMSE equals {bm_orig_rmse}")

    bm_orig_mae = np.sum(test_array[:, 5]) / len(test_array[:, 5])
    print(f"Bagged model on original data MAE equals {bm_orig_mae}")
    print('\n')

    orig_mse = np.sum(orig_array[:, 5]) / len(orig_array[:, 5])
    print(f"Original model on original data MSE equals {orig_mse}")

    orig_rmse = math.sqrt(np.sum(bm_array[:, 5]) / len(orig_array[:, 5]))
    print(f"Original model on original data RMSE equals {orig_rmse}")

    orig_mae = np.sum(orig_array[:, 6]) / len(orig_array[:, 6])
    print(f"Original model on original data MAE equals {orig_mae}")
    print('\n')

    test_mse = np.sum(test_array[:, 8]) / len(test_array[:, 8])
    print(f"Original vs bagged MSE equals {test_mse}")

    test_rmse = math.sqrt(np.sum(test_array[:, 8]) / len(test_array[:, 8]))
    print(f"Original vs bagged RMSE equals {test_rmse}")

    test_mae = np.sum(test_array[:, 9]) / len(test_array[:, 9])
    print(f"Original vs bagged MAE equals {test_mae}")

    iaroplot_diff_models(
        original_data[original_data.direction == select_direction].density,
        original_data[original_data.direction == select_direction].flow,
        original_data[original_data.direction == select_direction].car_proportion,
        bagged_data[bagged_data.direction == select_direction].centroid_density,
        bagged_data[bagged_data.direction == select_direction].centroid_flow,
        bagged_data[bagged_data.direction == select_direction].weight,
        bagged_model.x,
        bagged_model.y,
        bagged_model.get_frontier(),
        original_model.x,
        original_model.y,
        original_model.get_frontier(),
        "Figure - " + month + "_" + str(year) + "_tau_" + str(int(100*tau))
    )

    error_list = []
    error_list = [
        [bm_mse, bm_rmse, bm_mae],
        [bm_orig_mse, bm_orig_rmse, bm_orig_mae],
        [orig_mse, orig_rmse, orig_mae],
        [test_mse, test_rmse, test_mae],
        [max_density_bagged, max_flow_bagged, 0],
        [max_density_orig, max_flow_orig, 0]]

    end_time = time.perf_counter()
    print(f"Calculation of errors completed, it took {end_time-start_time:0.4f} seconds")

    return error_list


def predict_day(train_data: pd.DataFrame,
                test_data: pd.DataFrame, tau: float = 0.5, select_direction: int = 2):
    start_time = time.perf_counter()
    x_train = train_data[train_data.direction == select_direction].centroid_density
    y_train = train_data[train_data.direction == select_direction].centroid_flow
    w_train = train_data[train_data.direction == select_direction].weight

    x_test = test_data[test_data.direction == select_direction].centroid_density
    y_test = test_data[test_data.direction == select_direction].centroid_flow
    w_test = test_data[test_data.direction == select_direction].weight

    test_array = np.column_stack((x_train, y_train))

    train_model = wCQER.wCQR(y=y_train, x=x_train, w=w_train, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
    train_model.optimize(OPT_LOCAL)

    test_model = wCQER.wCQR(y=y_test, x=x_test, w=w_test, tau=tau, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
    test_model.optimize(OPT_LOCAL)

    """
    train_array order:
    0 - x_train, 1 - y_train, 2 - beta, 3 - alpha, 4 - residual, 5 - residual squared,
    6 - abs(residual)
    """

    train_array = np.column_stack((train_model.x, train_model.y))
    train_array.view('f8,f8').sort(order=['f0'], axis=0)
    flatten = train_model.get_beta().flatten()
    train_array = np.column_stack(
        (train_array, flatten))
    train_array = np.column_stack(
        (train_array, train_model.get_alpha()))
    train_array = np.column_stack(
        (train_array, train_model.get_residual()))
    train_array = np.column_stack(
        (train_array, np.square(train_array[:, 4])))
    train_array = np.column_stack(
        (train_array, abs(train_array[:, 4])))

    """
    orig_array order:
    0 - x_test, 1 - y_test, 2 - beta, 3 - alpha, 4 - residual, 5 - residual squared,
    6 - abs(residual)
    """

    orig_array = np.column_stack((test_model.x, test_model.y))
    orig_array.view('f8,f8').sort(order=['f0'], axis=0)
    flatten_orig = test_model.get_beta().flatten()
    orig_array = np.column_stack(
        (orig_array, flatten_orig))
    orig_array = np.column_stack(
        (orig_array, test_model.get_alpha()))
    orig_array = np.column_stack(
        (orig_array, test_model.get_residual()))
    orig_array = np.column_stack(
        (orig_array, np.square(orig_array[:, 4])))
    orig_array = np.column_stack(
        (orig_array, abs(orig_array[:, 4])))

    """
    test_array order: 0 - x_test, 1 - y_test, 2 - bagged_representor, 3 - bagged_residual,
    4 - bagged_residual squared, 5 - abs(residual),
    6 - orig_representor, 7 - residual btw orig and bm, 8 - 7 squared, 9 - abs(7)
    """
    test_array.view('f8,f8').sort(order=['f0'], axis=0)

    # Calculation of y_test_hat using representor function
    test_array = np.append(test_array, np.zeros((len(test_array), 1), dtype=float), axis=1)

    for i in range(len(test_array[:, 0])):
        test_array[i, 2] = representor(train_array[:, 3], train_array[:, 2], test_array[i, 0])

    test_array = np.column_stack(
        (test_array, test_array[:, 1] - test_array[:, 2]))

    test_array = np.column_stack(
        (test_array, np.square(test_array[:, 3])))

    test_array = np.column_stack(
        (test_array, abs(test_array[:, 3])))

    test_array = np.append(test_array, np.zeros((len(test_array), 1), dtype=float), axis=1)

    for i in range(len(test_array[:, 0])):
        test_array[i, 6] = representor(orig_array[:, 3], orig_array[:, 2], test_array[i, 0])

    test_array = np.column_stack(
        (test_array, test_array[:, 6] - test_array[:, 2]))

    test_array = np.column_stack(
        (test_array, np.square(test_array[:, 7])))

    test_array = np.column_stack(
        (test_array, abs(test_array[:, 7])))

    max_density_bagged = test_array[np.argmax(test_array, axis=0)[2]][0]
    max_flow_bagged = np.max(test_array[:, 2])

    max_density_orig = test_array[np.argmax(test_array, axis=0)[6]][0]
    max_flow_orig = np.max(test_array[:, 6])

    print(f"Max density of TRAIN model is {max_density_bagged}")
    print(f"Max flow of TRAIN model is {max_flow_bagged}")

    print(f"Max density of TEST model is {max_density_orig}")
    print(f"Max flow of TEST model is {max_flow_orig}")

    bm_mse = np.sum(train_array[:, 5]) / len(train_array[:, 5])
    print(f"TRAIN model on TRAIN data MSE equals {bm_mse}")

    bm_rmse = math.sqrt(np.sum(train_array[:, 5]) / len(train_array[:, 5]))
    print(f"TRAIN model on TRAIN data RMSE equals {bm_rmse}")

    bm_mae = np.sum(train_array[:, 6]) / len(train_array[:, 6])
    print(f"TRAIN model on TRAIN data MAE equals {bm_mae}")
    print('\n')

    bm_orig_mse = np.sum(test_array[:, 4]) / len(test_array[:, 4])
    print(f"TRAIN model on TEST data MSE equals {bm_orig_mse}")

    bm_orig_rmse = math.sqrt(np.sum(test_array[:, 4]) / len(test_array[:, 4]))
    print(f"TRAIN model on TEST data RMSE equals {bm_orig_rmse}")

    bm_orig_mae = np.sum(test_array[:, 5]) / len(test_array[:, 5])
    print(f"TRAIN model on TEST data MAE equals {bm_orig_mae}")
    print('\n')

    orig_mse = np.sum(orig_array[:, 5]) / len(orig_array[:, 5])
    print(f"Original model on TEST data MSE equals {orig_mse}")

    orig_rmse = math.sqrt(np.sum(train_array[:, 5]) / len(orig_array[:, 5]))
    print(f"TEST model on TEST data RMSE equals {orig_rmse}")

    orig_mae = np.sum(orig_array[:, 6]) / len(orig_array[:, 6])
    print(f"TEST model on TEST data MAE equals {orig_mae}")
    print('\n')

    test_mse = np.sum(test_array[:, 8]) / len(test_array[:, 8])
    print(f"TEST vs TRAIN MSE equals {test_mse}")

    test_rmse = math.sqrt(np.sum(test_array[:, 8]) / len(test_array[:, 8]))
    print(f"TEST vs TRAIN RMSE equals {test_rmse}")

    test_mae = np.sum(test_array[:, 9]) / len(test_array[:, 9])
    print(f"TEST vs TRAIN MAE equals {test_mae}")

    iaroplot_days(
        x_test,
        y_test,
        w_test,
        x_train,
        y_train,
        w_train,
        train_model.x,
        train_model.y,
        train_model.get_frontier(),
        test_model.x,
        test_model.y,
        test_model.get_frontier(),
        test_array,
        "Figure"
    )

    error_list = [
        [bm_mse, bm_rmse, bm_mae],
        [bm_orig_mse, bm_orig_rmse, bm_orig_mae],
        [orig_mse, orig_rmse, orig_mae],
        [test_mse, test_rmse, test_mae],
        [max_density_bagged, max_flow_bagged, 0],
        [max_density_orig, max_flow_orig, 0]]

    return error_list


def multi_tau_graph(x, y, w, tau):
    model = []
    for t in tau:
        trafmod = wCQER.wCQR(y=y, x=x, w=w, tau=t, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS)
        trafmod.optimize(OPT_LOCAL)
        model.append(trafmod)

    plt.clf()

    x = np.array(model[0].x).T[0]
    y = np.array(model[0].y).T
    yhat = np.array(model[0].get.frontier()).T
    data = (np.stack([x, y, yhat], axis=0)).T
    # sort
    data = data[np.argsort(data[:, 0])].T
    x1, y1, f1 = data[0], data[1], data[2]

    fig, axs = plt.subplots(1, 1, figsize=(15, 15), sharex='all', sharey='all')
    axs[0, 0].scatter(x1, y1, color="k", marker='o', label="TEST data")
    axs[0, 0].plot(x1, f1, color="g", label="0.20")

    yhat = np.array(model[1].get.frontier()).T
    data = (np.stack([x, y, yhat], axis=0)).T
    data = data[np.argsort(data[:, 0])].T
    x1, y1, f1 = data[0], data[1], data[2]
    axs[0, 0].plot(x1, f1, color="g", label="0.50")

    yhat = np.array(model[2].get.frontier()).T
    data = (np.stack([x, y, yhat], axis=0)).T
    data = data[np.argsort(data[:, 0])].T
    x1, y1, f1 = data[0], data[1], data[2]
    axs[0, 0].plot(x1, f1, color="g", label="0.75")

    yhat = np.array(model[3].get.frontier()).T
    data = (np.stack([x, y, yhat], axis=0)).T
    data = data[np.argsort(data[:, 0])].T
    x1, y1, f1 = data[0], data[1], data[2]
    axs[0, 0].plot(x1, f1, color="g", label="0.95")
    axs[0, 0].legend()

    plt.savefig("Fig")

    return

def fscalc(df: pd.DataFrame, aggregation_time_period=DEF_AGG_TIME_PER) -> pd.DataFrame:
    start_time = time.perf_counter()
    time_agg = pd.DataFrame()
    space_agg = pd.DataFrame()

    # Create the aggregation parametere based on aggregation_time_period
    df['aggregation'] = (df.hour * 60 + df.minute)/aggregation_time_period
    df = df.astype({'aggregation': int})

    # Aggregate flow and speed by time
    time_agg = df.groupby(['id', 'date', 'aggregation', 'direction'],
                          as_index=False).agg(smspeed=('speed', scipy.stats.hmean),
                                              flow=('cars', 'count'),
                                              cars=('cars', 'sum'),
                                              buses=('buses', 'sum'),
                                              trucks=('trucks', 'sum'))
    time_agg['hourlyflow'] = 60/aggregation_time_period * time_agg.flow
    time_agg['hourlycars'] = 60/aggregation_time_period * time_agg.cars
    time_agg['hourlybuses'] = 60/aggregation_time_period * time_agg.buses
    time_agg['hourlytrucks'] = 60/aggregation_time_period * time_agg.trucks
    time_agg['density'] = time_agg['hourlyflow'].div(
        time_agg['smspeed'].values)
    time_agg['seconds'] = time_agg['aggregation'] * 60 * aggregation_time_period
    time_agg['seconds'] = time_agg['seconds'].astype('float64')
    time_agg['time'] = pd.to_datetime(time_agg['seconds'], unit='s')
    time_agg['time'] = pd.Series([val.strftime("%H:%M") for val in time_agg['time']])

    """# Aggregate flow and speed by space and calculate density. Calculate the weights
    space_agg = time_agg.groupby(['id', 'date', 'aggregation', 'direction'],
                                 as_index=False).agg(qdivvsum=('qdivv', 'sum'),
                                                     flow=('hourlyflow', 'sum'),
                                                     cars=('hourlycars', 'sum'),
                                                     buses=('hourlybuses', 'sum'),
                                                     trucks=('hourlytrucks', 'sum'))
    space_agg['actual_flow'] = ti
    space_agg['space_mean_speed'] = 1/(space_agg.qdivvsum.div(space_agg.flow))
    space_agg['density'] = space_agg.flow.div(space_agg.space_mean_speed)
    space_agg['weight'] = float(1/len(space_agg))
    space_agg['car_proportion'] = space_agg.cars.div(space_agg.flow)
    space_agg['bus_proportion'] = space_agg.buses.div(space_agg.flow)
    space_agg['truck_proportion'] = space_agg.trucks.div(space_agg.flow)"""

    end_time = time.perf_counter()
    print(
        f"Aggregating data for modeling took {end_time-start_time:0.4f} seconds")

    return time_agg
