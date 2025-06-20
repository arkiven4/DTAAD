import torch, os
import pickle, os, sqlite3, copy, sklearn
from datetime import datetime
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime, timedelta

import src.models as models
 
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
   
def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return ((a - min_a) / (max_a - min_a + 0.0001)), min_a, max_a

def denormalize3(a_norm, min_a, max_a):
    return a_norm * (max_a - min_a + 0.0001) + min_a

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

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

def load_model(dataset, modelname, dims, retrainMode=False, testMode=False):
    model_class = getattr(models, modelname)
    model = model_class(dims).double()
    fname = f'checkpoints/{model.name}_{dataset}/model.ckpt'
    if os.path.exists(fname) and (not retrainMode or testMode):
        checkpoint = torch.load(fname, weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Creating new model: {model.name}")
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

def preprocessPD_loadData(df_sel, feature_set, min_a, max_a):
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

def update_statisticGlobal(mean_global, M2_global, n_total, mean_i, std_i, n_i):
    delta = mean_i - mean_global
    new_total = n_total + n_i
    mean_global += delta * n_i / new_total
    M2_global += std_i**2 * (n_i - 1)
    M2_global += delta**2 * n_total * n_i / new_total
    return mean_global, M2_global, new_total

def calcThres_oneModel(current_time, model_now, feature_set, current_loss):
    temp_severity = {}
    end_date = (current_time - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
    start_date = (current_time - timedelta(hours=12)).strftime('%Y-%m-%d %H:%M:%S')

    try:
        mean_data = fetch_between_dates(start_date, end_date, "db/adaptive_mean.db", model_now)[-1, 2:].astype(float)
        m2_data = fetch_between_dates(start_date, end_date, "db/adaptive_m2.db", model_now)[-1, 2:].astype(float)
        count_data = fetch_between_dates(start_date, end_date, "db/adaptive_count.db", model_now)[-1, 2:].astype(float)
    except:
        end_date = (current_time - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
        start_date = (current_time - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')

        mean_data = fetch_between_dates(start_date, end_date, "db/adaptive_mean.db", model_now)[-1, 2:].astype(float)
        m2_data = fetch_between_dates(start_date, end_date, "db/adaptive_m2.db", model_now)[-1, 2:].astype(float)
        count_data = fetch_between_dates(start_date, end_date, "db/adaptive_count.db", model_now)[-1, 2:].astype(float)

    for sensor_idx in range(current_loss.shape[-1]):
        feature_now_name = feature_set[sensor_idx]
        now_loss = current_loss[:, sensor_idx]

        std_global = (m2_data[sensor_idx] / (count_data[sensor_idx] - 1))**0.5
        loss_thre = mean_data[sensor_idx] + (2.102 * std_global) # 2.326 2.102
        loss_threVector = np.full(len(now_loss), loss_thre)
        loss_aboveThre = np.maximum(now_loss, loss_threVector)

        sever_comp1 = np.sum((now_loss > loss_thre)) / len(now_loss)
        sever_comp2 = np.mean(np.abs((loss_threVector - loss_aboveThre) / loss_threVector))
        if sever_comp2 > 1:
            sever_comp2 = 1

        severity_percentage = (sever_comp1 * sever_comp2) * 100
        temp_severity[feature_set[sensor_idx]] = round(float(severity_percentage), 2)

    return temp_severity, mean_data, m2_data, count_data

def calc_counterPercentage(threshold_percentages, feature_set, model_array):
    mean_severity_percentage = {}
    for feature_name in feature_set:
        mean_severity_percentage[feature_name] = {"count": 0, "percentage": 0}

    for modex_idx, values_pred in threshold_percentages.items():
        for name_feat, percentage in values_pred.items():
            mean_severity_percentage[name_feat]["count"] = mean_severity_percentage[name_feat]["count"] + 1 if percentage >= 20 else 0
            mean_severity_percentage[name_feat]["percentage"] = mean_severity_percentage[name_feat]["percentage"] + percentage

    for key, value in mean_severity_percentage.items():
        if mean_severity_percentage[key]['count'] >= 2:
            mean_severity_percentage[key]['count'] = round((mean_severity_percentage[key]['count'] / len(model_array)) * 100, 2)
            mean_severity_percentage[key]['percentage'] = round(float(mean_severity_percentage[key]['percentage'] // len(model_array)))
        else:
            mean_severity_percentage[key]['count'] = 0
            mean_severity_percentage[key]['percentage'] = 0

    #mean_severity_percentage = dict(sorted(mean_severity_percentage.items(), key=lambda item: item[1]['percentage'], reverse=True))
    # Find Which Model Have Highest Confidence
    counter_feature_plot = {}
    for index, value in mean_severity_percentage.items():
        higher_data = {"model": 0, "percentage": 0}
        for model_idx in threshold_percentages:
            if index in threshold_percentages[model_idx]:
                if higher_data["percentage"] <= threshold_percentages[model_idx][index]:
                    higher_data["model"] = model_idx
                    higher_data["percentage"] = threshold_percentages[model_idx][index]
        
        counter_feature_plot[index] = higher_data['model']

    return mean_severity_percentage, counter_feature_plot