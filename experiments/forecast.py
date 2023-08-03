import os
from os.path import join
import math
import logging
from typing import Callable, Optional, Union, Dict, Tuple, List

import gin
from fire import Fire
import numpy as np
import torch
import random
import warnings
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

from experiments.base import Experiment
from data.datasets import ForecastDataset
from models import get_model
from utils.checkpoint import Checkpoint
from utils.ops import default_device, to_tensor
from utils.losses import get_loss_fn
from utils.metrics import calc_metrics, evaluation_metrics
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import StandardScaler
from utils.model_efficiency import test_params_flop

warnings.simplefilter("ignore", category=RuntimeWarning)


class ForecastExperiment(Experiment):
    @gin.configurable()
    def instance(self,
                 model_type: str,
                 self_train: bool,
                 n_train_patient: int,
                 anchor_patient: int,
                 patient_ids: List,
                 self_train_subset_factor: float,
                 save_vals: Optional[bool] = True, ):
        if self_train:
            assert n_train_patient == 1
            train_patients = [anchor_patient]
            eval_patients = [anchor_patient]
        else:
            # get train patients and eval patients
            assert 0 < n_train_patient < len(patient_ids)
            idx = patient_ids.index(anchor_patient)
            train_patients = [patient_ids[(idx + i + 1) % len(patient_ids)] for i in range(n_train_patient)]
            eval_patients = [anchor_patient]

        train_set, train_loader, scaler = get_data(flag='train', patients=train_patients, subset_factor=None, scaler=None)
        val_set, val_loader, _ = get_data(flag='val', patients=train_patients, subset_factor=None, scaler=scaler)

        assert 0.0 <= self_train_subset_factor <= 1.0
        if self_train_subset_factor > 0.0:
            self_train_set, self_train_loader, _ = get_data(
                flag='train', patients=eval_patients, subset_factor=self_train_subset_factor, scaler=scaler
            )
            self_val_set, self_val_loader, _ = get_data(
                flag='val', patients=eval_patients, subset_factor=self_train_subset_factor, scaler=scaler
            )

        test_set, test_loader, _ = get_data(flag='test', patients=eval_patients, subset_factor=None, scaler=scaler)

        model = get_model(model_type,
                          lookback_len=train_set.lookback_len,
                          horizon_len=train_set.horizon_len,
                          datetime_feats=train_set.timestamps.shape[-1]).to(default_device())
        # test_params_flop(model, train_loader)
        checkpoint = Checkpoint(self.root)

        # train forecasting task
        if model_type != 'dummy':
            model = train(model, checkpoint, train_loader, val_loader, test_loader)
            if not self_train and self_train_subset_factor > 0.0:
                checkpoint.cross_init()
                model = train(model, checkpoint, self_train_loader, self_val_loader, test_loader)

        # testing
        val_metrics = validate(model, loader=val_loader, report_metrics=True)
        test_metrics = validate(model, loader=test_loader, report_metrics=True,
                                save_path=self.root if save_vals else None)

        # evaluation
        np.save(join(self.root, 'metrics.npy'), {'val': val_metrics, 'test': test_metrics})

        val_metrics = {f'ValMetric/{k}': v for k, v in val_metrics.items()}
        test_metrics = {f'TestMetric/{k}': v for k, v in test_metrics.items()}
        checkpoint.close({**val_metrics, **test_metrics})


@gin.configurable()
def get_optimizer(model: nn.Module,
                  lr: Optional[float] = 1e-3,
                  lambda_lr: Optional[float] = 1.,
                  weight_decay: Optional[float] = 1e-2) -> optim.Optimizer:
    group1 = []  # lambda
    group2 = []  # no decay
    group3 = []  # decay
    no_decay_list = ('bias', 'norm',)
    for param_name, param in model.named_parameters():
        if '_lambda' in param_name:
            group1.append(param)
        elif any([mod in param_name for mod in no_decay_list]):
            group2.append(param)
        else:
            group3.append(param)
    optimizer = optim.Adam([
        {'params': group1, 'weight_decay': 0, 'lr': lambda_lr, 'scheduler': 'cosine_annealing'},
        {'params': group2, 'weight_decay': 0, 'scheduler': 'cosine_annealing_with_linear_warmup'},
        {'params': group3, 'scheduler': 'cosine_annealing_with_linear_warmup'}
    ], lr=lr, weight_decay=weight_decay)
    return optimizer


@gin.configurable()
def get_scheduler(optimizer: optim.Optimizer,
                  T_max: int,
                  warmup_epochs: int,
                  eta_min: Optional[float] = 0.) -> optim.lr_scheduler.LambdaLR:
    scheduler_fns = []
    for param_group in optimizer.param_groups:
        scheduler = param_group['scheduler']
        if scheduler == 'none':
            fn = lambda T_cur: 1
        elif scheduler == 'cosine_annealing':
            lr = eta_max = param_group['lr']
            fn = lambda T_cur: (eta_min + 0.5 * (eta_max - eta_min) * (
                        1.0 + math.cos((T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
        elif scheduler == 'cosine_annealing_with_linear_warmup':
            lr = eta_max = param_group['lr']
            # https://blog.csdn.net/qq_36560894/article/details/114004799
            fn = lambda T_cur: T_cur / warmup_epochs if T_cur < warmup_epochs else (eta_min + 0.5 * (
                        eta_max - eta_min) * (1.0 + math.cos(
                (T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
        else:
            raise ValueError(f'No such scheduler, {scheduler}')
        scheduler_fns.append(fn)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fns)
    return scheduler


@gin.configurable()
def get_data(flag: bool,
             batch_size: int,
             patients: List,
             subset_factor: Optional = None,
             scaler: Optional = None) -> Tuple[ConcatDataset, DataLoader]:
    if flag in ('val', 'test'):
        shuffle = False
        drop_last = False
    elif flag == 'train':
        shuffle = True
        drop_last = True
    else:
        raise ValueError(f'no such flag {flag}')

    if scaler is None:
        scaler = ForecastDataset(flag, patient=patients[0], scaler=None).scaler
    dataset_list = []
    for p in patients:
        dataset_list.append(ForecastDataset(flag, patient=p, scaler=scaler))

    dataset = ConcatDataset(dataset_list)
    if subset_factor is not None:
        assert 0.0 <= subset_factor <= 1.0
        subset_count = int(len(dataset) * subset_factor)
        subset_indices = random.sample(list(range(len(dataset))), subset_count)
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    # add attributes to dataset
    dataset.lookback_len = dataset_list[0].lookback_len
    dataset.horizon_len = dataset_list[0].horizon_len
    dataset.timestamps = dataset_list[0].timestamps
    dataset.scaler = dataset_list[0].scaler
    dataset.inverse_transform = dataset_list[0].inverse_transform

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last)
    return dataset, data_loader, scaler


@gin.configurable()
def train(model: nn.Module,
          checkpoint: Checkpoint,
          train_loader: DataLoader,
          val_loader: DataLoader,
          test_loader: DataLoader,
          loss_name: str,
          epochs: int,
          clip: float) -> nn.Module:

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer=optimizer, T_max=epochs)
    training_loss_fn = get_loss_fn(loss_name)

    for epoch in range(epochs):
        train_loss = []
        model.train()
        for it, data in enumerate(train_loader):
            optimizer.zero_grad()

            x, y, x_time, y_time = map(to_tensor, data)
            forecast = model(x, x_time, y_time)

            if isinstance(forecast, tuple):
                # for models which require reconstruction + forecast loss
                loss = training_loss_fn(forecast[0], x) + \
                       training_loss_fn(forecast[1], y)
                raise
            else:
                loss = training_loss_fn(forecast, y)
                # loss = training_loss_fn(forecast[:, -1:, :], y[:, -1:, :])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            train_loss.append(loss.item())
            if (it + 1) % 100 == 0:
                logging.info(f"epochs: {epoch + 1}, iters: {it + 1} | training loss: {loss.item():.2f}")
        scheduler.step()

        train_loss = np.average(train_loss)
        val_loss = validate(model, loader=val_loader, loss_fn=training_loss_fn)
        test_loss = validate(model, loader=test_loader, loss_fn=training_loss_fn)

        scalars = {'Loss/Train': train_loss,
                   'Loss/Val': val_loss,
                   'Loss/Test': test_loss}
        checkpoint(epoch + 1, model, scalars=scalars)

        if checkpoint.early_stop:
            logging.info("Early stopping")
            break

    if epochs > 0:
        model.load_state_dict(torch.load(checkpoint.model_path))
    return model


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             loss_fn: Optional[Callable] = None,
             report_metrics: Optional[bool] = False,
             save_path: Optional[str] = None) -> Union[Dict[str, float], float]:
    model.eval()
    inputs = []
    preds = []
    trues = []
    inps = []
    total_loss = []
    # masks = []
    for it, data in enumerate(loader):
        x, y, x_time, y_time = map(to_tensor, data)

        if x.shape[0] == 1:
            # skip final batch if batch_size == 1
            # due to bug in torch.linalg.solve which raises error when batch_size == 1
            continue

        forecast = model(x, x_time, y_time)

        if report_metrics:
            inputs.append(x)
            preds.append(forecast)
            trues.append(y)
            if save_path is not None:
                inps.append(x)
        else:
            loss = loss_fn(forecast, y, reduction='none')
            total_loss.append(loss)

    if report_metrics:
        # (n, ph, 1)
        inputs = torch.cat(inputs, dim=0).detach().cpu().numpy()
        preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        trues = torch.cat(trues, dim=0).detach().cpu().numpy()

        def inverse_transform(x):
            if len(x.shape) == 3:
                n, h, _ = x.shape
                x = x.reshape(-1, 1)
                x = loader.dataset.inverse_transform(x)
                return x.reshape(n, h, 1)
            else:
                return loader.dataset.inverse_transform(x)
        inputs, preds, trues = inverse_transform(inputs), inverse_transform(preds), inverse_transform(trues)

        if save_path is not None:
            inps = torch.cat(inps, dim=0).detach().cpu().numpy()
            np.save(join(save_path, 'inps.npy'), inps)
            np.save(join(save_path, 'preds.npy'), preds)
            np.save(join(save_path, 'trues.npy'), trues)

        evaluations = evaluation_metrics(preds, trues, None)
        return evaluations

    total_loss = torch.cat(total_loss, dim=0).cpu()
    return np.average(total_loss)


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(ForecastExperiment)
