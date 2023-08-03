import datetime

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('ignore', Warning)

from pathlib import Path
from os.path import join
from tqdm import tqdm


train_sheets = ["Baseline", "3M", "6M", "9M", "12M", "15M", "18M", "21M", "24M", "27M", "30M", "33M"]
test_sheet = "36M"

train_months = [1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]
test_month = 36

patient_ids = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 27, 29, 30, 31, 32, 33, 35, 47]


def read_sheet(data_path, patient_id, month):
    if month == 1:
        sheet_name = "Baseline"
    else:
        sheet_name = f'{month}M'
    df = pd.read_excel(data_path, sheet_name=sheet_name)
    row_idx = df.index[df['Patient #'] == patient_id][0]
    row = df.iloc[row_idx]
    start_time = None
    glucose_res = []
    time_res = []
    for i in range(len(row)):
        if isinstance(row[i], datetime.time):
            start_time = row[i]
            start_time = datetime.datetime.combine(datetime.date(2000 + month // 12, month % 12 + 1, 1), start_time)
            time = start_time
            continue
        if not start_time is None:
            glucose = row[i]
            glucose_res.append(glucose)
            time_res.append(time)
            time = time + datetime.timedelta(minutes=5)
    # blocker of different sheets
    if len(glucose_res) == 0:
        return None
    for i in range(10):
        glucose_res.append(np.NAN)
        time_res.append(time_res[-1] + datetime.timedelta(minutes=5))

    df_res = pd.DataFrame(data=glucose_res, index=time_res, columns=['glucose'])
    df_res.index.name = 'date'
    return df_res


def main(data_path, out_path):
    Path(out_path).mkdir(exist_ok=True)
    for patient in patient_ids:
        train_dfs = []
        for train_month in train_months:
            temp = read_sheet(data_path, patient, month=train_month)
            if not temp is None:
                train_dfs.append(temp)
        train_df = pd.concat(train_dfs, axis=0)
        test_df = read_sheet(data_path, patient, month=test_month)
        train_df = train_df.interpolate(method="linear", limit=3)
        test_df = test_df.interpolate(method="linear", limit=3)
        train_df.to_csv(join(out_path, f'{patient}_train.csv'))
        test_df.to_csv(join(out_path, f'{patient}_test.csv'))
    return


if __name__ == '__main__':
    data_path = "storage/datasets/umt1dm/unprocessed_cgm_data.xlsx"
    out_path = "storage/datasets/umt1dm/preprocessed"
    main(data_path, out_path)
