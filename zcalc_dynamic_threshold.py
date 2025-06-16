import pickle
import os
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
import copy

from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader, TensorDataset

import sklearn
from scipy.signal import resample
from src.models import *
from src.utils import *
from main import  load_dataset, backprop

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return ((a - min_a) / (max_a - min_a + 0.0001)), min_a, max_a

def denormalize3(a_norm, min_a, max_a):
    return a_norm * (max_a - min_a + 0.0001) + min_a

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]  # cut
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])  # pad
        windows.append(w if 'DTAAD' in model.name or 'Attention' in model.name or 'TranAD' in model.name else w.view(-1))
    return torch.stack(windows)

def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double()
    fname = f'checkpoints/{model.name}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        checkpoint = torch.load(fname, weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        assert True
    return model

def filter_noise_es(df, alpha=0.4, reduction=False):
    import copy
    new_df = copy.deepcopy(df)
    
    for column in df:
        new_df[column] = df[column].ewm(alpha=alpha, adjust=False).mean()
    
    if reduction:
        return new_df[::len(df)]  # Adjust sparsity if needed
    else:
        return new_df

def wgn_pandas(df_withtime, snr, alpha=0.15, window_size=120):
    df_no_timestamp = df_withtime.drop(columns=['TimeStamp'])
    noisy_df = pd.DataFrame(index=df_no_timestamp.index, columns=df_no_timestamp.columns)

    for start in range(0, len(df_no_timestamp), window_size):
        window = df_no_timestamp.iloc[start:start + window_size]
        
        Ps = np.sum(np.power(window, 2), axis=0) / len(window)
        Pn = Ps / (np.power(10, snr / 10))

        noise = np.random.randn(*window.shape) * np.sqrt(Pn.values)
        noisy_window = window + (noise / 100)

        noisy_df.iloc[start:start + window_size] = noisy_window
    
    noisy_df.reset_index(drop=True, inplace=True)
    noisy_df = filter_noise_es(pd.DataFrame(noisy_df, columns=noisy_df.columns), alpha)

    df_timestamp = df_withtime['TimeStamp']
    df_timestamp.reset_index(drop=True, inplace=True)

    df_withtime = pd.concat([df_timestamp, noisy_df], axis=1)
    return df_withtime

def preprocessPD_loadData(df_sel):
    df_sel = wgn_pandas(df_sel, 30, alpha=0.15)

    df_timestamp = df_sel.iloc[:, 0]
    df_feature =  df_sel.iloc[:, 1:]
    df_feature = df_feature[feature_set]

    df_feature, _, _ = normalize3(df_feature, min_a, max_a)
    df_feature = df_feature.astype(float)

    test_loader = DataLoader(df_feature.values, batch_size=df_feature.shape[0])
    testD = next(iter(test_loader))
    testO = testD

    return testD, testO, df_timestamp, df_feature

def label_load(row):
   if row['Active Power'] < 1 and row['Governor speed actual'] < 1:
      return 'Shutdown'
   elif row['Active Power'] < 3 and row['Governor speed actual'] < 250:
      return 'Warming'
   elif row['Active Power'] < 3 and row['Governor speed actual'] > 250:
      return 'No Load'
   elif row['Active Power'] >= 1 and row['Active Power'] < 20 and row['Governor speed actual'] > 250:
      return 'Low Load'
   elif row['Active Power'] >= 20 and row['Active Power'] < 40 and row['Governor speed actual'] > 250:
      return 'Rough Zone'
   elif row['Active Power'] >= 40 and row['Active Power'] < 50 and row['Governor speed actual'] > 250:
      return 'Part Load'
   elif row['Active Power'] >= 50 and row['Active Power'] < 65 and row['Governor speed actual'] > 250:
      return 'Efficient Load'
   elif row['Active Power'] >= 65 and row['Governor speed actual'] > 250:
      return 'High Load'
   else:
      return 'Undefined'
   
def update_global_stats(mean_global, M2_global, n_total, mean_i, std_i, n_i):
    """
    Update global mean and variance accumulator M2 using a new window's stats.
    """
    delta = mean_i - mean_global
    new_total = n_total + n_i
    mean_global += delta * n_i / new_total
    M2_global += std_i**2 * (n_i - 1)
    M2_global += delta**2 * n_total * n_i / new_total

    return mean_global, M2_global, new_total

def init_db_timeconst(feature_set, db_name="masters_data.db", table_name="severity_trending"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create table if it does not exist
    columns = ", ".join([feature_name.replace(" ", "_") for feature_name in feature_set])
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            {columns}
        )
    """)

    conn.commit()
    conn.close()

def batch_timeseries_savedb(df_timestamps, data, feature_set, db_name="data.db", table_name="sensor_data"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Convert timestamps to ISO format
    timestamps = [pd.to_datetime(ts).isoformat() for ts in df_timestamps]
    
    # Build column names for features, replacing spaces with underscores
    feature_columns = ', '.join([feature_name.replace(" ", "_") for feature_name in feature_set])
    placeholders = ', '.join(['?' for _ in range(len(feature_set)+1)])  # 30 features + 1 timestamp
    
    # Prepare batch data
    batch_data = [(timestamps[i], *data[i]) for i in range(data.shape[0])]
    
    # Upsert using INSERT OR REPLACE (Ensure UNIQUE constraint on timestamp in your DB schema)
    sql = f"""
        INSERT OR REPLACE INTO {table_name} (timestamp, {feature_columns})
        VALUES ({placeholders})
    """
    
    cursor.executemany(sql, batch_data)
    conn.commit()
    conn.close()

def timeseries_savedb(df_timestamp, data, feature_set, db_name="data.db", table_name="sensor_data"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Generate timestamp
    timestamp = df_timestamp.isoformat()
    
    # Build column names for features, replacing spaces with underscores
    feature_columns = ', '.join([feature_name.replace(" ", "_") for feature_name in feature_set])
    placeholders = ', '.join(['?' for _ in range(len(feature_set))])
    
    # Upsert using INSERT OR REPLACE
    # Note: Your table must have a UNIQUE constraint on the timestamp column.
    sql = f"""
        INSERT OR REPLACE INTO {table_name} (timestamp, {feature_columns})
        VALUES (?, {placeholders})
    """
    cursor.execute(sql, (timestamp, *data))
    
    conn.commit()
    conn.close()

def fetch_between_dates(start_date, end_date, db_name="data.db", table_name="sensor_data"):
    start_date = start_date.replace(" ", "T")
    end_date = end_date.replace(" ", "T")
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT * FROM {table_name} WHERE timestamp BETWEEN ? AND ?
    """, (start_date, end_date))
    
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return np.array([])
    
    return np.array(rows)

def convert_timestamp(timestamp_str):
    dt = datetime.fromisoformat(timestamp_str)
    return pd.Timestamp(dt.strftime('%Y-%m-%d %H:%M:%S'))

feature_set = ['Active Power', 'Reactive Power', 'Governor speed actual', 'UGB X displacement', 'UGB Y displacement',
    'LGB X displacement', 'LGB Y displacement', 'TGB X displacement',
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

model_array = ["DTAAD", "MAD_GAN", "TranAD", "DAGMM", "USAD"] #["Attention", "DTAAD", "MAD_GAN", "TranAD", "DAGMM", "USAD", "OmniAnomaly"] # , CAE_M "GDN" MSCRED
model_thr = {}
for model_name in model_array:
    model_thr[model_name] = 0

for model_now in model_array:
    with open(f'loss_fold/{args.dataset}/{model_now}.pickle', 'rb') as handle:
        loss = pickle.load(handle)
    model_thr[model_now] = [np.percentile(loss[:, index], 99) for index in range(len(feature_set))]

measured_horizon = 60 * 2 * 1

df_data_withtime = pd.read_pickle("my_data_5thn_olah.pickle")
mask = (df_data_withtime['TimeStamp'] >= '2020-01-01 00:00:00')
df_data_withtime = df_data_withtime.loc[mask]

for column_name in df_data_withtime.columns:
    if column_name != 'Load_Type' and column_name != 'TimeStamp':
        df_data_withtime[column_name] = pd.to_numeric(df_data_withtime[column_name], downcast='float')
        
df_anomaly = pd.read_excel("shutdown_list.xlsx", 'Sheet2')
df_anomaly['Start Time'] = pd.to_datetime(df_anomaly['Start Time'])
df_anomaly['End Time'] = pd.to_datetime(df_anomaly['End Time'])
df_anomaly_unplaned = df_anomaly.copy()

mask = (df_anomaly_unplaned['Interal/External'] == 'Internal') & (df_anomaly_unplaned['Shutdown Type'] == 'Unplanned') & (df_anomaly_unplaned['Start Time'] >= '2020-01-01 00:00:00')
df_anomaly_unplaned = df_anomaly_unplaned.loc[mask]
df_anomaly_unplaned = df_anomaly_unplaned.reset_index(drop=True)
df_anomaly_unplaned = df_anomaly_unplaned.drop(df_anomaly_unplaned.index[[2]])
df_anomaly_unplaned = df_anomaly_unplaned.reset_index(drop=True)
df_anomaly_unplaned

# Calc Initial
mean_global = {}
M2_global = {}
n_total = {}
for model_now in model_array:
    mean_global[model_now] = {}
    M2_global[model_now] = {}
    n_total[model_now] = {}
    for feature_name in feature_set:
        mean_global[model_now][feature_name] = 0.0
        M2_global[model_now][feature_name] = 0.0
        n_total[model_now][feature_name] = 0.0

interval_gap = 30
end_date_filter = pd.to_datetime('2020-06-30 06:15:00') #- timedelta(minutes=5)
start_trend_filter = pd.to_datetime('2020-01-01 06:15:00') #- timedelta(days=120)
current_end_window = start_trend_filter

df_timestamp_last = np.datetime64('2012-04-28T04:16:00.000000000')
total_steps = int((end_date_filter - current_end_window).total_seconds() // (interval_gap * 60)) + 1
for _ in tqdm(range(total_steps), desc="Progress"):
    start_date_window = current_end_window - timedelta(minutes=measured_horizon)
    mask = (df_data_withtime['TimeStamp'] > start_date_window.strftime('%Y-%m-%d %H:%M:%S')) & (
        df_data_withtime['TimeStamp'] <= current_end_window.strftime('%Y-%m-%d %H:%M:%S'))
    df_sel = df_data_withtime.loc[mask]
    df_additional = df_sel[['Grid Selection']].copy()
    df_additional = df_additional.astype(float)
    df_sel = df_sel[['TimeStamp'] + feature_set]

    load_label = df_sel.apply(label_load, axis=1).value_counts()
    no_load = load_label.get('No Load', 0)
    shutdown = load_label.get('Shutdown', 0)
    total = load_label.sum()
    bad_pct = (no_load + shutdown) / total
    if bad_pct > 0.05:  # More than 5%
        continue

    testD, testO, df_timestamp, df_feature = preprocessPD_loadData(df_sel)
    df_timestamp = df_timestamp.values

    for model_now in model_array:
        model = load_model(model_now, testO.shape[1])
        model.eval()
        torch.zero_grad = True

        if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
            testD_now = convert_to_windows(testD, model)
        else:
            testD_now = testD    

        loss, y_pred = backprop(0, model, testD_now, testO, None, None, training=False)

        for i_loss in range(loss.shape[-1]):
            mean_i = np.mean(loss[:, i_loss])
            std_i = np.std(loss[:, i_loss], ddof=1)
            n_i = len(loss[:, i_loss])

            feature_now_name = feature_set[i_loss]

            mean_global[model_now][feature_now_name], M2_global[model_now][feature_now_name], n_total[model_now][feature_now_name] = update_global_stats(
                mean_global[model_now][feature_now_name], M2_global[model_now][feature_now_name], n_total[model_now][feature_now_name], mean_i, std_i, n_i
            )

    # DONT REMOVE THIS
    df_timestamp_last = df_timestamp[-1]
    current_end_window += timedelta(minutes=interval_gap)

init_db_timeconst(feature_set, "db/original_data.db", "original_data")
init_db_timeconst(['Grid Selection'], "db/original_data.db", "additional_original_data")
for model_name in model_array:
    init_db_timeconst(feature_set, "db/pred_data.db", model_name)
    init_db_timeconst(feature_set, "db/threshold_mean.db", model_name)
    init_db_timeconst(feature_set, "db/threshold_m2.db", model_name)
    init_db_timeconst(feature_set, "db/threshold_count.db", model_name)

interval_gap = 30
end_date_filter = pd.to_datetime('2023-12-28 00:00:00') #- timedelta(minutes=5)
start_trend_filter = pd.to_datetime('2020-06-30 06:20:00') #- timedelta(days=120)
current_end_window = start_trend_filter

df_timestamp_last = np.datetime64('2020-06-30T06:10:00.000000000')
total_steps = int((end_date_filter - current_end_window).total_seconds() // (interval_gap * 60)) + 1
for _ in tqdm(range(total_steps), desc="Progress"):
    start_date_window = current_end_window - timedelta(minutes=measured_horizon)
    mask = (df_data_withtime['TimeStamp'] > start_date_window.strftime('%Y-%m-%d %H:%M:%S')) & (
        df_data_withtime['TimeStamp'] <= current_end_window.strftime('%Y-%m-%d %H:%M:%S'))
    df_sel = df_data_withtime.loc[mask].iloc[:120, :]
    df_additional = df_sel[['Grid Selection']].copy()
    df_additional = df_additional.astype(float)
    df_sel = df_sel[['TimeStamp'] + feature_set]

    load_label = df_sel.apply(label_load, axis=1).value_counts()
    no_load = load_label.get('No Load', 0)
    shutdown = load_label.get('Shutdown', 0)
    total = load_label.sum()
    bad_pct = (no_load + shutdown) / total

    testD, testO, df_timestamp, df_feature = preprocessPD_loadData(df_sel)
    df_timestamp = df_timestamp.dt.floor("min").values[:120]

    ypred_models = {} 
    calc_stats = {}
    for model_now in model_array:
        model = load_model(model_now, testO.shape[1])
        model.eval()
        torch.zero_grad = True
        if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
            testD_now = convert_to_windows(testD, model)
        else:
            testD_now = testD
        loss, y_pred = backprop(0, model, testD_now, testO, None, None, training=False)
        ypred_models[model_now] = denormalize3(y_pred, min_a, max_a)

        calc_stats[model_now] = {}
        for idx_feat in range(loss.shape[-1]):
            feature_now_name = feature_set[idx_feat]

            calc_stats[model_now][feature_now_name] = True
            temp_std = (M2_global[model_now][feature_now_name] / (n_total[model_now][feature_now_name] - 1))**0.5 if n_total[model_now][feature_now_name] > 1 else 0.0
            temp_mean = mean_global[model_now][feature_now_name]

            thres_bool1 = loss[:, idx_feat] > temp_mean
            thres_percentage1 = (thres_bool1.sum() / thres_bool1.shape[0]) * 100

            thres_bool2 = loss[:, idx_feat] > temp_mean + (temp_std)
            thres_percentage2 = (thres_bool2.sum() / thres_bool2.shape[0]) * 100

            # thres_bool2 = loss[:, idx_feat] < temp_mean - (temp_std)
            # thres_percentage2 = (thres_bool2.sum() / thres_bool2.shape[0]) * 100

            if thres_percentage1 >= 20 or thres_percentage2 >= 5: # or thres_percentage2 >= 50:
                calc_stats[model_now][feature_now_name]  = False

    mask = df_timestamp > df_timestamp_last
    df_feature = denormalize3(df_feature, min_a, max_a)
    df_feature = df_feature[mask].values
    df_additional = df_additional[mask].values
    df_timestamp = df_timestamp[mask]
    for model_now in model_array:
        ypred_models[model_now] = ypred_models[model_now][mask]
    
    batch_timeseries_savedb(df_timestamp, trunc(df_feature, decs=2), feature_set, "db/original_data.db", "original_data")
    batch_timeseries_savedb(df_timestamp, trunc(df_additional, decs=2), ['Grid Selection'], "db/original_data.db", "additional_original_data")
    for idx_model, (model_name) in enumerate(model_array):
        batch_timeseries_savedb(df_timestamp, trunc(ypred_models[model_name], decs=2), feature_set, "db/pred_data.db", model_name) 

    if bad_pct < 0.001:  # Less than 0.1%  -> Dont Calc Mean and STD
        for model_now in model_array:
                for i_loss in range(loss.shape[-1]):
                    feature_now_name = feature_set[idx_feat]
                    if calc_stats[model_now][feature_now_name]:
                        mean_i = np.mean(loss[:, i_loss])
                        std_i = np.std(loss[:, i_loss], ddof=1)
                        n_i = len(loss[:, i_loss])
        
                        feature_now_name = feature_set[i_loss]

                        mean_global[model_now][feature_now_name], M2_global[model_now][feature_now_name], n_total[model_now][feature_now_name] = update_global_stats(
                            mean_global[model_now][feature_now_name], M2_global[model_now][feature_now_name], n_total[model_now][feature_now_name], mean_i, std_i, n_i)

    df_timestampi = pd.to_datetime(df_timestamp[0])
    for model_idx, model_name in enumerate(model_array):
        timeseries_savedb(df_timestampi, trunc(np.array(list(mean_global[model_name].values())), decs=6), feature_set, "db/threshold_mean.db", model_name) 
        timeseries_savedb(df_timestampi, trunc(np.array(list(M2_global[model_name].values())), decs=6), feature_set, "db/threshold_m2.db", model_name) 
        timeseries_savedb(df_timestampi, trunc(np.array(list(n_total[model_name].values())), decs=1), feature_set, "db/threshold_count.db", model_name) 

    # DONT REMOVE THIS
    df_timestamp_last = df_timestamp[-1]
    current_end_window += timedelta(minutes=interval_gap)