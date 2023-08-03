from xml.etree import ElementTree

import pandas as pd
import numpy as np
import datetime


def round_minute(date_string, round2min):
    # convert the date to round2min intervals.
    # For simplicity rounded backwards. i.e. 0,1,2,3,4 -> 0
    round_min2 = round2min
    date = datetime.datetime.strptime(date_string, "%d-%m-%Y %H:%M:%S")
    new_min = ((date.minute//round_min2) * round_min2)
    date = date.replace(minute=int(new_min))
    date = date.replace(second=0)
    return date


def get_cgm(root, round2min):
    glucose = []
    glucose_ts = []
    for type_tag in root.findall('glucose_level/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts')
        ts = round_minute(ts, round2min)
        glucose.append(value)
        glucose_ts.append(ts)
    dataframe_cgm = pd.DataFrame(data=glucose, index=glucose_ts, columns=["glucose"])
    return dataframe_cgm


# constant column
def get_basal(root, round2min):
    basal = []
    basal_ts = []
    for type_tag in root.findall('basal/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts')
        ts = round_minute(ts, round2min)
        basal.append(value)
        basal_ts.append(ts)
    dataframe_basal = pd.DataFrame(data=basal, index=basal_ts, columns=["basal"])
    return dataframe_basal


def get_temp_basal(root, round2min):
    temp_basal = []
    temp_basal_ts = []
    temp_basal_end = []
    for type_tag in root.findall('temp_basal/event'):
        value = type_tag.get('value')
        ts = type_tag.get('ts_begin')
        ts = round_minute(ts, round2min)
        ts_end = type_tag.get('ts_end')
        ts_end = round_minute(ts_end, round2min)

        temp_basal_end.append(ts_end)
        temp_basal.append(value)
        temp_basal_ts.append(ts)
    dataframe_temp_basal = pd.DataFrame(data=np.array([temp_basal, temp_basal_end]).T, index=temp_basal_ts, columns=["temp_basal", "temp_basal_end"])
    return dataframe_temp_basal



# single value column
def get_bolus(root, round2min):
    bolus = []
    bolus_ts = []
    bolus_end = []

    for type_tag in root.findall('bolus/event'):
        value = type_tag.get('dose')
        ts = type_tag.get('ts_begin')
        ts = round_minute(ts, round2min)
        ts_end = type_tag.get('ts_end')
        ts_end = round_minute(ts_end, round2min)
        bolus_end.append(ts_end)

        bolus.append(value)
        bolus_ts.append(ts)
    dataframe_bolus = pd.DataFrame(data=np.array([bolus, bolus_end]).T, index=bolus_ts, columns=["bolus", "bolus_end"])
    return dataframe_bolus


def get_meal(root, round2min):
    meal = []
    meal_ts = []
    for type_tag in root.findall('meal/event'):
        carbs = type_tag.get('carbs')
        ts = type_tag.get('ts')
        ts = round_minute(ts, round2min)
        meal.append(carbs)
        meal_ts.append(ts)
    dataframe_meal = pd.DataFrame(data=meal, index=meal_ts, columns=["carbs"])
    return dataframe_meal


def parse_xml(raw_path):
    root = ElementTree.parse(raw_path).getroot()
    round2min = 5

    cgm = get_cgm(root, round2min)
    basal = get_basal(root, round2min)
    temp_basal = get_temp_basal(root, round2min)
    bolus = get_bolus(root, round2min)
    meal = get_meal(root, round2min)

    cgm_start, cgm_end = cgm.index[0], cgm.index[-1]

    covariates = [cgm, basal, temp_basal, bolus, meal]
    def get_global_df(covariates):
        global_start = min([x.index[0] for x in covariates if len(x) > 0])
        global_end = max([x.index[-1] for x in covariates if len(x) > 0])
        return pd.DataFrame(
            index=pd.date_range(
                global_start, global_end, periods=(global_end-global_start)//pd.Timedelta(5,"minutes")+1
            )
        )
    df = get_global_df(covariates)
    for df_ in covariates:
        df = df.join(df_, how="left")
    df.index.name = "date"
    df = df.astype(
        {
            "glucose": "float",
            "basal": "float",
            "temp_basal": "float",
            "bolus": "float",
            "carbs": "float"
        }
    )
    df = df.groupby(df.index).max()
    df["temp_basal_end"] = pd.to_datetime(df["temp_basal_end"])
    df["bolus_end"] = pd.to_datetime(df["bolus_end"])
    return df, cgm_start, cgm_end
