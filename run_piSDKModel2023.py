import pickle
import os
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
import copy
import time

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

import sys
import clr

sys.path.append(r'C:\Program Files (x86)\PIPC\AF\PublicAssemblies\4.0')  
clr.AddReference('OSIsoft.AFSDK')

from OSIsoft.AF import *  
from OSIsoft.AF.PI import *
from OSIsoft.AF.Search import * 
from OSIsoft.AF.Asset import *  
from OSIsoft.AF.Data import *  
from OSIsoft.AF.Time import *  
from OSIsoft.AF.UnitsOfMeasure import *

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
    fname = f'checkpoints/{modelname}_{args.dataset}/model.ckpt'
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

def percentage2severity(value):
    return (
        1 if 0 <= value < 5 else
        2 if 5 <= value < 20 else
        3 if 20 <= value < 40 else
        4 if 40 <= value < 75 else
        5 if 75 <= value <= 100 else
        6
    )

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

def calcThres_allModel(threshold_percentages, temp_ypreds, model_array, testD, testO):
    for idx_model, model_now in enumerate(model_array):
        model = load_model(model_now, testO.shape[1])
        model.eval()
        torch.zero_grad = True

        if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 
                            'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
            testD_now = convert_to_windows(testD, model)
        else:
            testD_now = testD

        loss, y_pred = backprop(0, model, testD_now, testO, None, None, training=False)

        threshold_pass = {}
        for idx_feat in range(loss.shape[-1]):
            thres_bool = loss[:, idx_feat] > model_thr[model_now][idx_feat]
            threshold_pass[feature_set[idx_feat]] = (thres_bool.sum() / thres_bool.shape[0]) * 100
        
        threshold_percentages[idx_model] = threshold_pass
        temp_ypreds[idx_model] = denormalize3(y_pred, min_a, max_a)

    return threshold_percentages, temp_ypreds

def calc_counterPercentageTrending(threshold_percentages):
    counter_feature = {}
    for modex_idx, values_pred in threshold_percentages.items():
        for name_feat, percentage in values_pred.items():
            if name_feat in counter_feature:
                if percentage > 5.0:
                    counter_feature[name_feat]["count"] = counter_feature[name_feat]["count"] + 1
                    counter_feature[name_feat]["percentage"] = counter_feature[name_feat]["percentage"] + percentage
            else:
                counter_feature[name_feat] = {"count": 1, "percentage": percentage}

    for key, value in counter_feature.items():
        counter_feature[key]['count'] = (counter_feature[key]['count'] / len(model_array)) * 100
        if counter_feature[key]['count'] >= 20.0:
            counter_feature[key]['severity'] = percentage2severity(counter_feature[key]['percentage'] // len(model_array))
            counter_feature[key]['percentage'] = (counter_feature[key]['percentage'] // len(model_array))
        else:
            counter_feature[key]['severity'] = 1
            counter_feature[key]['percentage'] = 0.0

    return counter_feature

def parse_recorded_events(recorded):
    parsed_events = []
    for event in recorded:
        timestamp_str = str(event.Timestamp.LocalTime)  # Convert .NET DateTime to string
        timestamp = pd.to_datetime(timestamp_str)  # Convert to pandas Timestamp
        value = event.Value
        parsed_events.append((timestamp, value))
    return pd.DataFrame(parsed_events, columns=['Timestamps', 'Values'])

def getdf_piserver(piServer, pi_tag, time_list):
    timerange = AFTimeRange(time_list[0], time_list[1])
    master_pd = "";
    for i in range(len(pi_tag)):
        tag = PIPoint.FindPIPoint(piServer, pi_tag[i])
        value_resp = parse_recorded_events(tag.InterpolatedValues(timerange, AFTimeSpan.Parse('1m'), '', False))
        if i == 0:
            value_resp['Timestamps'] = pd.to_datetime(value_resp['Timestamps'])
            master_pd = value_resp
        else:
            master_pd = pd.concat([master_pd, value_resp['Values']], axis=1, join='inner')

    master_pd = master_pd.values
    master_pd = pd.DataFrame(data=master_pd, columns=['TimeStamp'] + feature_set)
    df_sel = master_pd.iloc[-120:, :]
    df_sel = df_sel.reset_index(drop=True)

    for column_name in df_sel.columns:
        if column_name != 'Load_Type' and column_name != 'TimeStamp':
            df_sel[column_name] = pd.to_numeric(df_sel[column_name], downcast='float')
   
    return df_sel

def init_db(feature_set, db_name="masters_data.db", table_name="severity_trending"):
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

def trend_savedb(data, feature_set, db_name="data.db", table_name="sensor_data"):
    #if len(data) == 30:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    timestamp = datetime.now().replace(second=0, microsecond=0).isoformat()
    cursor.execute(f"""
        INSERT INTO {table_name} (timestamp, {', '.join([feature_name.replace(" ", "_") for feature_name in feature_set])})
        VALUES (?, {', '.join(['?' for _ in range(len(feature_set))])})
    """, (timestamp, *data))
    
    conn.commit()
    conn.close()

def timeseries_savedb(df_timestamp, data, feature_set, db_name="data.db", table_name="sensor_data"):
    #if len(data) == 30:
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

def batch_timeseries_savedb(df_timestamps, data, feature_set, db_name="data.db", table_name="sensor_data"):
    # if data.shape[1] != 30:
    #     raise ValueError("Data must have exactly 30 features")
    
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

def fetch_between_dates(start_date, end_date, db_name="data.db", table_name="sensor_data"):
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

feature_tag_mapping = {
    'Stator winding temperature 13': 'U-LGS1-TI-81104A-AI',
    'Stator winding temperature 14': 'U-LGS1-TI-81104B-AI',
    'Stator winding temperature 15': 'U-LGS1-TI-81104C-AI',
    'Surface Air Cooler Air Outlet Temperature': 'U-LGS1-TI-81104D-AI',
    'Surface Air Cooler Water Inlet Temperature': 'U-LGS1-TI-81104E-AI',
    'Surface Air Cooler Water Outlet Temperature': 'U-LGS1-TI-81104F-AI',
    'Stator core temperature': 'U-LGS1-TI-81104G-AI',
    'UGB metal temperature': 'U-LGS1-TI-81104H-AI',
    'UGB oil temperature': 'U-LGS1-TI-81104I-AI',
    'LGB metal temperature 1': 'U-LGS1-TI-81104J-AI',
    'LGB metal temperature 2': 'U-LGS1-TI-81104K-AI',
    'LGB oil temperature': 'U-LGS1-TI-81104L-AI',
    'Governor speed actual': 'U-LGS1-SI-81101-AI',
    'UGB X displacement': 'U-LGS1-UGB-X-PK-PK-70-AI',
    'UGB Y displacement': 'U-LGS1-UGB-Y-PK-PK-340-AI',
    'LGB X displacement': 'U-LGS1-GB-X-PK-PK-70-AI',
    'LGB Y displacement': 'U-LGS1-LGB-Y-PK-PK-340-AI',
    'TGB X displacement': 'U-LGS1-TGB-X-PK-PK-270-AI',
    'TGB Y displacement': 'U-LGS1-TGB-Y-PK-PK-340-AI',
    'Active Power': 'U-LGS1-Active-Power-AI',
    'Reactive Power': 'U-LGS1-Reactive-Power-AI',
    'Opening Wicked Gate': 'U-LGS1-ZT-81101-AI',
    'UGB Oil Contaminant': 'U-LGS1-AY-81103B-AI',
    'Gen Thrust Bearing Oil Contaminant': 'U-LGS1-AY-81103C-AI',
    'Gen Voltage Phase 1': 'U-LGS1-EI_81151A_MV-AI',
    'Gen Voltage Phase 2': 'U-LGS1-EI_81151B_MV-AI',
    'Gen Voltage Phase 3': 'U-LGS1-EI_81151C_MV-AI',
    'Gen Current Phase 1': 'U-LGS1-II_81152A_MV-AI',
    'Gen Current Phase 2': 'U-LGS1-II_81152B_MV-AI',
    'Gen Current Phase 3': 'U-LGS1-II_81152C_MV-AI',
    'Penstock Flow': 'U-LGS1-FI-81101-AI',
    'Turbine flow': 'U-LGS1-FIT-431-AI',
    'UGB cooling water flow': 'U-LGS1-FIT-81103A-AI',
    'LGB cooling water flow': 'U-LGS1-FIT-81103B-AI',
    'Generator cooling water flow': 'U-LGS1-FIT-81103C-AI',
    'Governor Penstock Pressure': 'U-LGS1-PI-81101-AI',
    'Penstock pressure': 'U-LGS1-PT-81150-AI'
}

tag_array = [feature_tag_mapping[feature] for feature in feature_set]

with open('normalize_2023.pickle', 'rb') as handle:
    normalize_obj = pickle.load(handle)
    min_a, max_a = normalize_obj['min_a'], normalize_obj['max_a']

model_array = ["Attention", "DTAAD", "MAD_GAN", "TranAD", "DAGMM", "USAD", "OmniAnomaly"] # , CAE_M "GDN" MSCRED
model_thr = {}
for model_name in model_array:
    model_thr[model_name] = 0

for model_now in model_array:
    with open(f'loss_fold/{args.dataset}/{model_now}.pickle', 'rb') as handle:
        loss = pickle.load(handle)
    model_thr[model_now] = [np.percentile(loss[:, index], 99) for index in range(len(feature_set))]

measured_horizon = 60 * 2 * 1

init_db_timeconst(feature_set, "db/original_data.db", "original_data")
init_db_timeconst(feature_set, "db/severity_trendings.db", "severity_trendings")
init_db_timeconst(feature_set, "db/severity_trendings.db", "original_sensor")
for model_name in model_array:
    init_db_timeconst(feature_set, "db/pred_data.db", model_name)
    init_db_timeconst(feature_set, "db/threshold_data.db", model_name)

piServers = PIServers()
piServer = piServers["PTI-PI"]                                                    #Write PI Server Name
piServer.Connect(False)                                                             #Connect to PI Server
print ('Connected to server: ' + "PTI-PI")

count = 0
#df_timestamp_last = np.datetime64('2020-04-28T04:16:00.000000000')
conn = sqlite3.connect("db_data/original_data.db")
cursor = conn.cursor()
cursor.execute(f"""SELECT * FROM original_data order by rowid desc LIMIT 1""")
rows = cursor.fetchall()
conn.close()
df_timestamp_last = np.datetime64(np.array(rows)[:, 1][0]) 

while True:
    print("Executing task... " + str(count))
    
    now_time = datetime.utcnow()
    start_time = now_time - timedelta(minutes=measured_horizon)
    time_list = [start_time.strftime('%Y-%m-%d %H:%M:%S'), now_time.strftime('%Y-%m-%d %H:%M:%S')]
    df_sel = getdf_piserver(piServer, tag_array, time_list)

    threshold_percentages = {}
    temp_ypreds = {}

    testD, testO, df_timestamp, df_feature = preprocessPD_loadData(df_sel)
    threshold_percentages, temp_ypreds = calcThres_allModel(threshold_percentages, temp_ypreds, model_array, testD, testO)

    df_feature = denormalize3(df_feature, min_a, max_a)
    df_feature_mean = trunc(np.mean(df_feature.values, axis=0), decs=2)

    df_feature = resample(df_feature, 20, axis=0)
    df_timestamp = df_timestamp.dt.floor("min")[::6].values
    for i in range(len(model_array)):
        temp_ypreds[i] = resample(temp_ypreds[i], 20, axis=0)

    mask = df_timestamp > df_timestamp_last
    df_feature = df_feature[mask]
    df_timestamp = df_timestamp[mask]
    for i in range(len(model_array)):
        temp_ypreds[i] = temp_ypreds[i][mask]

    # Trending
    counter_feature_trd = calc_counterPercentageTrending(threshold_percentages)
    trend_data = np.array([counter_feature_trd[key]['percentage'] for key in counter_feature_trd])
    
    batch_timeseries_savedb(df_timestamp, trunc(df_feature, decs=2), feature_set, "db/original_data.db", "original_data")
    for idx_model, (model_name) in enumerate(model_array):
        batch_timeseries_savedb(df_timestamp, trunc(temp_ypreds[idx_model], decs=2), feature_set, "db/pred_data.db", model_name) 

    df_timestampi = pd.to_datetime(df_timestamp[-1])
    for model_idx, model_name in enumerate(model_array):
        timeseries_savedb(df_timestampi, trunc(np.array(list(threshold_percentages[model_idx].values())), decs=2), feature_set, "db/threshold_data.db", model_name) 

    timeseries_savedb(df_timestampi, trend_data, feature_set, "db/severity_trendings.db", "severity_trendings") 
    timeseries_savedb(df_timestampi, df_feature_mean, feature_set, "db/severity_trendings.db", "original_sensor") 

    # DONT REMOVE THIS
    df_timestamp_last = df_timestamp[-1]
    
    count = count + 1
    time.sleep(600)