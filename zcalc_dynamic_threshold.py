import pickle
import os
import sqlite3
import copy
import sklearn
import pandas as pd
import numpy as np
from tqdm import tqdm

from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from scipy.signal import resample

from src.models import *
from src.utils import *
from main import load_dataset, backprop
import src.commons as commons

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

measured_horizon = 60 * 2 * 1
interval_gap = 30
model_array = ["DTAAD", "DAGMM", "USAD"]

feature_set = ['Active Power', 'Reactive Power', 'Governor speed actual', 'UGB X displacement', 
               'UGB Y displacement', 'LGB X displacement', 'LGB Y displacement', 'TGB X displacement',
               'TGB Y displacement', 'Stator winding temperature 13',
               'Stator winding temperature 14', 'Stator winding temperature 15',
               'Surface Air Cooler Air Outlet Temperature',
               'Surface Air Cooler Water Inlet Temperature',
               'Surface Air Cooler Water Outlet Temperature',
               'Stator core temperature', 'UGB metal temperature',
               'LGB metal temperature 1', 'LGB metal temperature 2',
               'LGB oil temperature', 'Penstock Flow', 'Turbine flow',
               'UGB cooling water flow', 'LGB cooling water flow',
               'Generator cooling water flow', 'Governor Penstock Pressure',
               'Penstock pressure', 'Opening Wicked Gate', 'UGB Oil Contaminant',
               'Gen Thrust Bearing Oil Contaminant']

with open('normalize_2023.pickle', 'rb') as handle:
    normalize_obj = pickle.load(handle)
    min_a, max_a = normalize_obj['min_a'], normalize_obj['max_a']

commons.init_db_timeconst(feature_set, "db/original_data.db", "original_data")
# init_db_timeconst(['Grid Selection'], "db/original_data.db", "additional_original_data")
commons.init_db_timeconst(feature_set, "db/severity_trendings.db", "severity_trendings")
commons.init_db_timeconst(feature_set, "db/severity_trendings.db", "original_sensor")
for model_name in model_array:
    commons.init_db_timeconst(feature_set, "db/pred_data.db", model_name)
    commons.init_db_timeconst(feature_set, "db/threshold_data.db", model_name)

    commons.init_db_timeconst(feature_set, "db/adaptive_mean.db", model_name)
    commons.init_db_timeconst(feature_set, "db/adaptive_m2.db", model_name)
    commons.init_db_timeconst(feature_set, "db/adaptive_count.db", model_name)

# df_data_withtime = pd.read_pickle("/run/media/fourier/Data2/Pras/Vale/time-series-autoencoder/my_data_5thn_olah.pickle")
df_data_withtime = pd.read_csv("Data20212025.csv", parse_dates=['TimeStamp'])
mask = (df_data_withtime['TimeStamp'] >= '2020-01-01 00:00:00')
df_data_withtime = df_data_withtime.loc[mask].sort_values("TimeStamp").reset_index(drop=True)
for column_name in df_data_withtime.columns:
    if column_name != 'Load_Type' and column_name != 'TimeStamp':
        df_data_withtime[column_name] = pd.to_numeric(
            df_data_withtime[column_name], downcast='float')

df_anomaly_unplaned = (
    pd.read_excel("/run/media/fourier/Data2/Pras/Vale/time-series-autoencoder/shutdown_list.xlsx", sheet_name='Sheet2')
    .assign(
        **{'Start Time': lambda df: pd.to_datetime(df['Start Time']),
           'End Time': lambda df: pd.to_datetime(df['End Time'])}
    )
    .query("`Interal/External` == 'Internal' and `Shutdown Type` == 'Unplanned' and `Start Time` >= '2020-01-01'")
    .reset_index(drop=True)
)
df_anomaly_unplaned = df_anomaly_unplaned.drop(df_anomaly_unplaned.index[[2]]).reset_index(drop=True)

if True:
    mean_global = {}
    M2_global = {}
    n_total = {}
    for model_now in model_array:
        with open(f'loss_fold/{args.dataset}/{model_now}_statistics.pickle', 'rb') as handle:
            now_statistics = pickle.load(handle)
        mean_global[model_now] = np.array(now_statistics["mean"])
        M2_global[model_now] = np.array(now_statistics["m2"])
        n_total[model_now] = np.array(now_statistics["count"])

    df_timestamp_last = np.datetime64('2012-04-28T04:16:00.000000000')
    end_date_filter = pd.to_datetime('2020-06-30 06:15:00')  # - timedelta(minutes=5)
    start_trend_filter = pd.to_datetime('2020-01-01 06:15:00')  # - timedelta(days=120)
    current_end_window = start_trend_filter

    total_steps = int((end_date_filter - current_end_window).total_seconds() // (interval_gap * 60)) + 1
    for step_prog in tqdm(range(total_steps), desc="Progress"):
        start_date_window = current_end_window - timedelta(minutes=measured_horizon)
        mask = (df_data_withtime['TimeStamp'] > start_date_window.strftime('%Y-%m-%d %H:%M:%S')) & (df_data_withtime['TimeStamp'] <= current_end_window.strftime('%Y-%m-%d %H:%M:%S'))
        df_sel = df_data_withtime.loc[mask]
        df_sel = df_sel[['TimeStamp'] + feature_set]

        load_label = df_sel.apply(commons.label_load, axis=1).value_counts()
        if (load_label.get('No Load', 0) + load_label.get('Shutdown', 0)) > 0: # load_label.get('No Load', 0) + load_label.get('Shutdown', 0)
            # DONT REMOVE THIS
            df_timestamp_last = df_sel['TimeStamp'].values[-1]
            current_end_window += timedelta(minutes=interval_gap)
            continue

        testD, testO, df_timestamp, df_feature = commons.preprocessPD_loadData(df_sel, feature_set, min_a, max_a)

        df_timestamp = df_timestamp.dt.floor("min")[::6].values[:20]
        mask = df_timestamp > df_timestamp_last
        df_timestamp = df_timestamp[mask]
        df_timestampi = pd.to_datetime(df_timestamp[0])

        for model_now in model_array:
            if step_prog > 50:
                end_date = (current_end_window - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
                start_date = (current_end_window - timedelta(hours=12)).strftime('%Y-%m-%d %H:%M:%S')

                try:
                    mean_data = commons.fetch_between_dates(start_date, end_date, "db/adaptive_mean.db", model_now)[-1, 2:].astype(float)
                    m2_data = commons.fetch_between_dates(start_date, end_date, "db/adaptive_m2.db", model_now)[-1, 2:].astype(float)
                    count_data = commons.fetch_between_dates(start_date, end_date, "db/adaptive_count.db", model_now)[-1, 2:].astype(float)
                    
                    mean_global[model_now] = mean_data
                    M2_global[model_now] = m2_data
                    n_total[model_now] = count_data
                except:
                    end_date = (current_end_window - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
                    start_date = (current_end_window - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')

                    mean_data = commons.fetch_between_dates(start_date, end_date, "db/adaptive_mean.db", model_now)[-1, 2:].astype(float)
                    m2_data = commons.fetch_between_dates(start_date, end_date, "db/adaptive_m2.db", model_now)[-1, 2:].astype(float)
                    count_data = commons.fetch_between_dates(start_date, end_date, "db/adaptive_count.db", model_now)[-1, 2:].astype(float)
                    
                    mean_global[model_now] = mean_data
                    M2_global[model_now] = m2_data
                    n_total[model_now] = count_data

            model = commons.load_model(args.dataset, model_now, testO.shape[1], args.retrain, args.test)
            model.eval()
            torch.zero_grad = True
            if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
                testD_now = commons.convert_to_windows(testD, model)
            else:
                testD_now = testD
            loss, y_pred = backprop(0, model, testD_now, testO, None, None, training=False)

            for i_loss in range(loss.shape[-1]):
                na_feature = feature_set[i_loss]

                mean_i = np.mean(loss[:, i_loss])
                std_i = np.std(loss[:, i_loss], ddof=1)
                n_i = len(loss[:, i_loss])

                mean_prev = mean_global[model_now][i_loss]
                M2_prev = M2_global[model_now][i_loss]
                n_prev = n_total[model_now][i_loss]

                mean_new, M2_new, n_new = commons.update_statisticGlobal(mean_prev, M2_prev, n_prev, mean_i, std_i, n_i)
                mean_global[model_now][i_loss] = mean_new
                M2_global[model_now][i_loss] = M2_new
                n_total[model_now][i_loss] = n_new

            commons.timeseries_savedb(df_timestampi, commons.trunc(mean_global[model_now], decs=6), feature_set, "db/adaptive_mean.db", model_now)
            commons.timeseries_savedb(df_timestampi, commons.trunc(M2_global[model_now], decs=6), feature_set, "db/adaptive_m2.db", model_now)
            commons.timeseries_savedb(df_timestampi, commons.trunc(n_total[model_now], decs=1), feature_set, "db/adaptive_count.db", model_now)

        # DONT REMOVE THIS
        df_timestamp_last = df_timestamp[-1]
        current_end_window += timedelta(minutes=interval_gap)


df_timestamp_last = np.datetime64('2020-06-30T06:15:00')
end_date_filter = pd.to_datetime('2025-06-02 09:35:00')  # - timedelta(minutes=5)
start_trend_filter = pd.to_datetime('2020-07-01 06:15:00')  # - timedelta(days=120)
current_end_window = start_trend_filter

total_steps = int((end_date_filter - current_end_window).total_seconds() // (interval_gap * 60)) + 1
for _ in tqdm(range(total_steps), desc="Progress"):
    start_date_window = current_end_window - timedelta(minutes=measured_horizon)
    mask = (df_data_withtime['TimeStamp'] > start_date_window.strftime('%Y-%m-%d %H:%M:%S')) & (df_data_withtime['TimeStamp'] <= current_end_window.strftime('%Y-%m-%d %H:%M:%S'))
    df_sel = df_data_withtime.loc[mask].iloc[:120, :]
    df_sel = df_sel[['TimeStamp'] + feature_set]

    load_label = df_sel.apply(commons.label_load, axis=1).value_counts()
    bad_pct = (load_label.get('No Load', 0) +  load_label.get('Shutdown', 0)) / load_label.sum()

    testD, testO, df_timestamp, df_feature = commons.preprocessPD_loadData(df_sel, feature_set, min_a, max_a)

    threshold_percentages = {}
    ypred_models = {}
    calc_stats = {}
    for model_now in model_array:
        model = commons.load_model(args.dataset, model_now, testO.shape[1], args.retrain, args.test)
        model.eval()
        torch.zero_grad = True
        if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
            testD_now = commons.convert_to_windows(testD, model)
        else:
            testD_now = testD
        loss, y_pred = backprop(0, model, testD_now, testO, None, None, training=False)

        ypred_models[model_now] = commons.denormalize3(y_pred, min_a, max_a)
        threshold_percentages[model_now], mean_data, m2_data, count_data = commons.calcThres_oneModel(current_end_window, model_now, feature_set, loss)

        calc_stats[model_now] = { "mean_glob": mean_data, "m2_glob": m2_data, "count_glob": count_data, "status_feat": {}}
        for idx_feat in range(loss.shape[-1]):
            feature_now_name = feature_set[idx_feat]

            calc_stats[model_now]["status_feat"][feature_now_name] = True
            if threshold_percentages[model_now][feature_now_name] >= 5.0:
                calc_stats[model_now]["status_feat"][feature_now_name] = False

    df_feature = commons.denormalize3(df_feature, min_a, max_a)
    df_feature_mean = commons.trunc(np.mean(df_feature.values, axis=0), decs=2)

    df_feature = resample(df_feature, 20, axis=0)
    # df_additional = resample(df_additional, 20, axis=0)
    df_timestamp = df_timestamp.dt.floor("min")[::6].values[:20]
    for model_now in model_array:
        ypred_models[model_now] = resample(ypred_models[model_now], 20, axis=0)

    min_len = min(len(df_timestamp), len(df_feature), *[len(ypred_models[m]) for m in model_array])
    df_timestamp = df_timestamp[:min_len]
    df_feature = df_feature[:min_len]
    for model_now in model_array:
        ypred_models[model_now] = ypred_models[model_now][:min_len]

    mask = df_timestamp > df_timestamp_last
    df_feature = df_feature[mask]
    df_timestamp = df_timestamp[mask]
    for model_now in model_array:
        ypred_models[model_now] = ypred_models[model_now][mask]

    commons.batch_timeseries_savedb(df_timestamp, commons.trunc(df_feature, decs=2), feature_set, "db/original_data.db", "original_data")
    # batch_timeseries_savedb(df_timestamp, trunc(df_additional, decs=2), ['Grid Selection'], "db/original_data.db", "additional_original_data")
    for idx_model, (model_name) in enumerate(model_array):
        commons.batch_timeseries_savedb(df_timestamp, commons.trunc(ypred_models[model_name], decs=2), feature_set, "db/pred_data.db", model_name)

    df_timestampi = pd.to_datetime(df_timestamp[-1])
    counter_feature_trd, _ = commons.calc_counterPercentage(threshold_percentages, feature_set, model_array)
    trend_data = np.array([counter_feature_trd[key]['percentage']for key in counter_feature_trd]).astype(np.float64)
    commons.timeseries_savedb(df_timestampi, trend_data, feature_set, "db/severity_trendings.db", "severity_trendings")
    commons.timeseries_savedb(df_timestampi, df_feature_mean, feature_set, "db/severity_trendings.db", "original_sensor")
    for model_idx, model_name in enumerate(model_array):
        commons.timeseries_savedb(df_timestampi, commons.trunc(np.array(list(
            threshold_percentages[model_name].values())), decs=2), feature_set, "db/threshold_data.db", model_name)

    if bad_pct == 0.0:
        for model_now in model_array:
            for i_loss in range(loss.shape[-1]):
                na_feature = feature_set[i_loss]
                if calc_stats[model_now]["status_feat"][na_feature] == True:
                    mean_i = np.mean(loss[:, i_loss])
                    std_i = np.std(loss[:, i_loss], ddof=1)
                    n_i = len(loss[:, i_loss])

                    mean_glob = calc_stats[model_now]["mean_glob"][i_loss]
                    m2_glob = calc_stats[model_now]["m2_glob"][i_loss]
                    count_glob = calc_stats[model_now]["count_glob"][i_loss]

                    mean_glob, m2_glob, count_glob = commons.update_statisticGlobal(mean_glob, m2_glob, count_glob, mean_i, std_i, n_i)
                    calc_stats[model_now]["mean_glob"][i_loss] = mean_glob
                    calc_stats[model_now]["m2_glob"][i_loss] = m2_glob
                    calc_stats[model_now]["count_glob"][i_loss] = count_glob

    for model_idx, model_name in enumerate(model_array):
        commons.timeseries_savedb(df_timestampi, commons.trunc(calc_stats[model_name]["mean_glob"], decs=6), feature_set, "db/adaptive_mean.db", model_name)
        commons.timeseries_savedb(df_timestampi, commons.trunc(calc_stats[model_name]["m2_glob"], decs=6), feature_set, "db/adaptive_m2.db", model_name)
        commons.timeseries_savedb(df_timestampi, commons.trunc(calc_stats[model_name]["count_glob"], decs=1), feature_set, "db/adaptive_count.db", model_name)

    # DONT REMOVE THIS
    df_timestamp_last = df_timestamp[-1]
    current_end_window += timedelta(minutes=interval_gap)
