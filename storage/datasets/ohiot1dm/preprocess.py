""" Preprocess OhioT1DM dataset.
Missing data present in glucose recording only.
Linear interpolation for up to 15 minutes gap.
"""

import pandas as pd
import warnings
warnings.simplefilter('ignore', Warning)

from pathlib import Path
from os.path import join
from tqdm import tqdm

from storage.datasets.ohiot1dm.xml_utils import parse_xml


def preprocess(raw_path):
    df, cgm_start, cgm_end = parse_xml(raw_path)

    # overwrite basal
    df['basal'] = df['basal'].fillna(method='ffill')  # fill with previous value
    for i in range(len(df)):
        if not pd.isna(df['temp_basal'][i]):
            temp_basal_end = df['temp_basal_end'][i]
            for j in range(i, len(df)):
                if df.index[j] <= temp_basal_end:
                    df["basal"][j] = df['temp_basal'][i]
                else:
                    break
    df = df.drop(["temp_basal", "temp_basal_end"], axis=1)

    # long act insulin
    df['bolus'] = df['bolus'].fillna(0.0)
    for i in range(0, len(df)):
        if not pd.isna(df['bolus_end'][i]) and df['bolus_end'][i] > df.index[i]:
            bolus_end = df['bolus_end'][i]
            for j in range(i, len(df)):
                if df.index[j] > bolus_end:
                    break
            df["bolus"][i:j] = round(df['bolus'][i]/(j-i), 2)
    df = df.drop(["bolus_end"], axis=1)

    df['carbs'] = df['carbs'].fillna(0.0)

    return df[(df.index >= cgm_start) & (df.index <= cgm_end)]


def pid2rawpath(ohiot1dm_path, patient_id, split):
    old = [559, 563, 570, 575, 588, 591]
    new = [540, 544, 552, 567, 584, 596]
    assert split in ["training", "testing"]
    assert patient_id in old + new
    if patient_id in old:
        return join(ohiot1dm_path, f"OhioT1DM-{split}", f"{patient_id}-ws-{split}.xml")
    else:
        return join(ohiot1dm_path, f"OhioT1DM-2-{split}", f"{patient_id}-ws-{split}.xml")


def main(ohiot1dm_path):
    out_path = join(ohiot1dm_path, "preprocessed")
    Path(out_path).mkdir(exist_ok=True)

    # preprocess(pid2rawpath(ohiot1dm_path, 540, "training"))
    for patient_id in tqdm([559, 563, 570, 575, 588, 591] + [540, 544, 552, 567, 584, 596]):
        df_train = preprocess(pid2rawpath(ohiot1dm_path, patient_id, "training"))
        df_test = preprocess(pid2rawpath(ohiot1dm_path, patient_id, "testing"))
        df_train = df_train.interpolate(method="linear", limit=3)
        df_test = df_test.interpolate(method="linear", limit=3)
        df_train.to_csv(join(out_path, f"{patient_id}_train.csv"))
        df_test.to_csv(join(out_path, f"{patient_id}_test.csv"))


if __name__ == '__main__':
    ohiot1dm_path = "/storage/datasets/ohiot1dm"
    main(ohiot1dm_path)
