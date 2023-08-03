import os
from os.path import join
from typing import Optional, List, Tuple, Callable

import gin
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from utils.time_features import get_time_features
from utils.filters import hyperglycemia, hypoglycemia
from utils.ops import parse_datetime


@gin.configurable()
class ForecastDataset(Dataset):
    def __init__(self,
                 flag: str,
                 patient: int,
                 horizon_len: int,
                 scale: bool,
                 # cross_learn: bool,
                 data_path: str,
                 root_path: Optional[str] = 'storage/datasets',
                 # features: Optional[str] = 'S',
                 target: Optional[str] = 'OT',
                 lookback_len: Optional[int] = None,
                 lookback_aux_len: Optional[int] = 0,
                 lookback_mult: Optional[float] = None,
                 time_features: Optional = [],
                 normalise_time_features: Optional = True,
                 # postprandial_len: Optional[int] = 48,
                 # daytime_start: Optional[int] = 8,
                 # daytime_end: Optional[int] = 22,
                 scaler: Optional[StandardScaler] = None
    ):
        """
        :param flag: train/val/test flag
        :param horizon_len: number of time steps in forecast horizon
        :param scale: performs standard scaling
        :param data_path: relative (to root_path) path to data file (.csv)
        :param cross_learn: treats multivariate time series as multiple univar time series
        :param root_path: path to datasets folder
        :param features: multivar (M) or univar (S) forecasting
        :param target: name of target variable for univar forecasting (features=S)
        :param lookback_len: number of time steps in lookback window
        :param lookback_aux_len: number of time steps to append to y from the lookback window
        (for models with decoders which requires initialisation)
        :param lookback_mult: multiplier to decide lookback window length
        # :param scenario: allday/postprandial/nocturnal/diurnal
        """
        assert flag in ('train', 'val', 'test'), \
            f"flag should be one of (train, val, test)"
        # assert features in ('M', 'S'), \
        #     f"features should be one of (M: multivar, S: univar)"
        assert (lookback_len is not None) ^ (lookback_mult is not None), \
            f"only 'lookback_len' xor 'lookback_mult' should be specified"
        # assert scenario in ('all', 'postprandial', 'nocturnal', 'diurnal')

        self.flag = flag
        self.patient = patient
        self.lookback_len = int(horizon_len * lookback_mult) if lookback_mult is not None else lookback_len
        self.lookback_aux_len = lookback_aux_len
        self.horizon_len = horizon_len
        self.scale = scale  # True
        # self.cross_learn = cross_learn  # False
        self.data_path = data_path
        self.root_path = root_path
        # self.features = features  # "M"
        self.target = target  # "glucose"
        self.time_features = time_features  # []
        self.normalise_time_features = normalise_time_features  # True
        # self.postprandial_len = postprandial_len
        # self.daytime_start = daytime_start
        # self.daytime_end = daytime_end

        # self.n_dims = None
        self.scaler = scaler
        self.data = None
        self.covariates_data = None
        # self.data_x = None
        # self.data_y = None
        self.timestamps = None
        # self.masks = None  # dict

        # self.hyper_events = None
        # self.hypo_events = None
        self.load_data()

    def get_boarder(self, index):
        x_start = index
        x_end = x_start + self.lookback_len
        y_start = x_end - self.lookback_aux_len
        y_end = y_start + self.lookback_aux_len + self.horizon_len
        return x_start, x_end, y_start, y_end

    def filter_missing(self,
                       df_data: np.array,
                       indices: List) -> list:
        res = []
        for index in indices:
            x_start, x_end, y_start, y_end = self.get_boarder(index)
            if not np.isnan(np.min(df_data[x_start: y_end])):
                res.append(index)
        return res

    def load_data(self):
        df_train_val_raw = pd.read_csv(join(self.root_path, self.data_path, f"{self.patient}_train.csv"))
        num_train = int(len(df_train_val_raw) * 0.8)
        num_val = len(df_train_val_raw) - num_train
        df_train_raw = df_train_val_raw.iloc[:num_train]
        df_val_raw = df_train_val_raw.iloc[num_train:]
        df_test_raw = pd.read_csv(join(self.root_path, self.data_path, f"{self.patient}_test.csv"))
        df_train_data = df_train_raw[[self.target]]

        if self.flag == "train":
            df_raw = df_train_raw
        elif self.flag == "val":
            df_raw = df_val_raw
        elif self.flag == "test":
            df_raw = df_test_raw
        else:
            raise ValueError
        cols = list(df_train_raw.columns)
        cols.remove('date')
        cols.remove(self.target)
        # df_train_covariates_data = df_train_raw[cols]

        df_data = df_raw[[self.target]]
        # df_covariates_data = df_raw[cols]

        indices = list(range(len(df_data)-self.lookback_len-self.horizon_len+1))
        indices = self.filter_missing(df_data.values, indices)
        self.indices = indices

        # scale
        if self.scaler is None:
            self.scaler = StandardScaler()
        # self.covariates_scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(df_train_data.values)
            # self.covariates_scaler.fit(df_train_covariates_data.values)
            self.data = self.scaler.transform(df_data.values)
            # self.covariates_data = self.covariates_scaler.transform(df_covariates_data.values)
        else:
            self.data = df_data.values

        self.timestamps = get_time_features(pd.to_datetime(df_raw.date.values),
                                            normalise=self.normalise_time_features,
                                            features=self.time_features)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # if self.cross_learn:
        #     dim_idx = idx // self.n_time_samples
        #     dim_slice = slice(dim_idx, dim_idx + 1)
        #     idx = idx % self.n_time_samples
        # else:
        #     dim_slice = slice(None)

        x_start, x_end, y_start, y_end = self.get_boarder(self.indices[idx])

        x = self.data[x_start:x_end, slice(None)]
        y = self.data[y_start:y_end, slice(None)]
        x_time = self.timestamps[x_start:x_end]
        y_time = self.timestamps[y_start:y_end]

        # x_time = np.concatenate(
        #     (x_time, self.covariates_data[x_start:x_end]), axis=1
        # )
        # y_time = np.concatenate(
        #     (y_time, self.covariates_data[y_start:y_end]), axis=1
        # )

        assert not np.isnan(np.min(x)) and not np.isnan(np.min(y))
        return x, y, x_time, y_time

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
