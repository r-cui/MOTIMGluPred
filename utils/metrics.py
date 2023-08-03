import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, matthews_corrcoef


def rse(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def corr(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def mae(pred, true):
    return np.mean(np.abs(pred - true))


def mse(pred, true):
    return np.mean((pred - true) ** 2)


def rmse(pred, true):
    return np.sqrt(mse(pred, true))


def mape(pred, true):
    return np.mean(np.abs((pred - true) / true))


def mspe(pred, true):
    return np.mean(np.square((pred - true) / true))


def calc_metrics(pred, true):
    # total
    res = {
        'rmse': rmse(pred, true),
        'mape': mape(pred, true)
    }
    # position wise
    for i in range(pred.shape[1]):
        # for i in [5, 11]:
        pred_temp = pred[:, i]
        true_temp = true[:, i]
        res.update({
            f'rmse{i}': rmse(pred_temp, true_temp),
            f'mape{i}': mape(pred_temp, true_temp),
        })

    return res

    # return {'mae': mae(pred, true),
    #         'mse': mse(pred, true),
    #         'rmse': rmse(pred, true),
    #         'mape': mape(pred, true),
    #         'mspe': mspe(pred, true)}


##############################################
# def ravel(conf_mat):
#     """ Export the metrics from a 2d confusion matrix.
#     Args:
#         conf_mat: np.array(
#             [[int, int],
#              [int, int]]
#         )
#     """
#     tn, fp, fn, tp = conf_mat.ravel()
#     se = round(tp / (tp + fn + 1e-6), 2)
#     sp = round(tn / (tn + fp + 1e-6), 2)
#     fa = round(fp / (tp + fp + 1e-6), 2)
#     mcc = round((tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + 1e-6), 2)
#
#     return tn, fp, fn, tp, se, sp, fa, mcc


def trend_polarity(pred, true, delta=1):
    def trend(y):
        ph = y.shape[1]
        assert ph > (2 * delta)
        num = ph - 2 * delta
        term1, term2 = y[:, delta: delta + num].copy(), y[:, delta: delta + num].copy()
        for i in range(1, delta + 1):
            term1 = term1 + y[:, delta + i: delta + num + i]
        for i in range(1, delta + 1):
            term2 = term2 + y[:, delta - i: delta + num - i]
        return ((term1 - term2) > 0.0).flatten()

    pred_trend = trend(pred)
    true_trend = trend(true)
    acc = accuracy_score(true_trend, pred_trend) * 100
    return {f"trend_{delta}_acc": acc}


def glycemia(pred, true, flag):
    if flag == "hyper":
        condition = lambda x: x >= 180
    elif flag == "hypo":
        condition = lambda x: x <= 70
    else:
        raise

    pred = condition(pred).flatten()
    true = condition(true).flatten()

    mcc = matthews_corrcoef(true, pred) * 100
    return {
        f"{flag}_mcc": mcc
    }


def glycemia_onset(pred, true, flag):
    if flag == "hyper":
        condition = lambda x: x >= 180
    elif flag == "hypo":
        condition = lambda x: x <= 70
    else:
        raise

    def detect_event(y):
        _, ph, _ = y.shape
        y_bool = condition(y)
        shift0 = y_bool[:, :-2]
        shift1 = y_bool[:, 1:-1]
        shift2 = y_bool[:, 2:]
        event_onset = (shift0 < shift1) & (shift1 & shift2)
        res_onset = np.any(event_onset, axis=1)
        res_time = np.argmax(event_onset, axis=1)
        return res_onset, res_time

    pred_onset, pred_time = detect_event(pred)
    true_onset, true_time = detect_event(true)

    mutual_positive = (pred_onset & true_onset).flatten()
    pred_time = pred_time[mutual_positive]
    true_time = true_time[mutual_positive]
    time = mae(pred_time, true_time) * 5  # CGM sampling rate
    mcc = matthews_corrcoef(true_onset, pred_onset) * 100
    return {
        f"{flag}_onset_mcc": mcc,
        f"{flag}_onset_time": time,
    }


def evaluate_metric(pred, true, mask=None):
    """
    :param pred: (n, horizon, 1)
    :param true: (n, horizon, 1)
    :param masks: (n, horizon, 1)
    :return:
    """
    res = dict()
    if mask is None:
        mask = np.ones(pred.shape)

    # mask select rows
    mask = np.all(mask, axis=1).squeeze(1)
    count = np.sum(mask)
    if count == 0:
        return res
    pred, true = pred[mask], true[mask]
    res.update(calc_metrics(pred, true))
    res.update(trend_polarity(pred, true, delta=1))
    res.update(trend_polarity(pred, true, delta=2))
    res.update(glycemia(pred, true, "hyper"))
    res.update(glycemia(pred, true, "hypo"))
    res.update(glycemia_onset(pred, true, "hyper"))
    res.update(glycemia_onset(pred, true, "hypo"))
    return res


def evaluation_metrics(pred, true, masks=None):
    """
    :param pred: (n, horizon, 1)
    :param true: (n, horizon, 1)
    :param masks: (n, horizon, n_masks)
    :return:
    """
    # res = dict()
    #
    # def add_prefix(d, prefix):
    #     for k in d.keys():
    #         res[f"{prefix}/{k}"] = d[k]
    #
    # add_prefix(evaluate_metric(pred, true, mask=None), "all")
    # for i in range(masks.shape[-1]):
    #     prefix = SCENARIOS[i]
    #     add_prefix(evaluate_metric(pred, true, masks[:, :, i:i + 1]), prefix)
    # return res

    return evaluate_metric(pred, true, None)
