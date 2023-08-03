import os
import shutil
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pprint import pprint
from pathlib import Path
from os.path import join


np.seterr(all="ignore")


def collect_as_parent(experiment_dir, parent_keys):
    def dict2str(d):
        return ",".join([f"{k}={d[k]}" for k in d.keys()])

    for instance in os.listdir(experiment_dir):
        settings = instance.split(",")
        all_keys = [a.split("=")[0] for a in settings]
        settings = {
            x: y for x, y in zip([a.split("=")[0] for a in settings], [a.split("=")[1] for a in settings])
        }
        parent_path = join(experiment_dir, dict2str({x: settings[x] for x in parent_keys}))
        Path(parent_path).mkdir(exist_ok=True)
        shutil.move(join(experiment_dir, instance), parent_path)
        new_name = dict2str({x: settings[x] for x in all_keys if x not in parent_keys})
        os.rename(join(parent_path, instance), join(parent_path, new_name))


def collect_as_child(experiment_dir, collect_keys):
    def dict2str(d):
        return ",".join([f"{k}={d[k]}" for k in d.keys()])

    for instance in os.listdir(experiment_dir):
        settings = instance.split(",")
        all_keys = [a.split("=")[0] for a in settings]
        parent_keys = [x for x in all_keys if x not in collect_keys]
        settings = {
            x: y for x, y in zip([a.split("=")[0] for a in settings], [a.split("=")[1] for a in settings])
        }
        parent_path = join(experiment_dir, dict2str({x: settings[x] for x in parent_keys}))
        Path(parent_path).mkdir(exist_ok=True)
        shutil.move(join(experiment_dir, instance), parent_path)
        new_name = dict2str({x: settings[x] for x in collect_keys})
        os.rename(join(parent_path, instance), join(parent_path, new_name))


def eval_dir(dir, mode="stats"):
    """ Evaluate the logs.
    Must be the lowest hierarchy collect_directories() were applied.
    """
    assert mode in ["stats", "scores"]
    result_list = []
    for instance in os.listdir(dir):
        scores = np.load(os.path.join(dir, instance, "metrics.npy"), allow_pickle=True).item()
        result_list.append(scores["test"])

    res = dict()
    if mode == "stats":
        for x in result_list[0].keys():
            if "mape" in x:
                res[x] = "{:.2f} ({:.2f})".format(np.nanmean([y[x] for y in result_list if x in y.keys()]) * 100,
                                                  np.nanstd([y[x] for y in result_list if x in y.keys()]) * 100)
            else:
                res[x] = "{:.2f} ({:.2f})".format(np.nanmean([y[x] for y in result_list if x in y.keys()]),
                                                  np.nanstd([y[x] for y in result_list if x in y.keys()]))
    elif mode == "scores":
        for x in result_list[0].keys():
            if "mape" in x:
                res[x] = [y[x] * 100 for y in result_list if x in y.keys()]
            else:
                res[x] = [y[x] for y in result_list if x in y.keys()]
    else:
        raise
    return res


def result_csv(log_dir):
    keys = [
        "rmse", "rmse5", "rmse11",
        "mape", "mape5", "mape11",
        "trend_1_acc", "trend_2_acc",
        "hyper_mcc",  # "hyper_onset_mcc",
        "hypo_mcc",  # "hypo_onset_mcc",
    ]
    dfs = []
    for model_name in sorted(os.listdir(log_dir)):
        dir = os.path.join(log_dir, model_name)
        model_res = eval_dir(dir, mode="stats")
        dfs.append(pd.DataFrame(model_res, index=[model_name]))
    df = pd.concat(dfs)
    pprint(df[keys])
    return df[keys]


def violin(log_dir, fig, axs, row_i):
    keys = [
        "rmse", #"rmse5", "rmse11",
        "mape", #"mape5", "mape11",
        "trend_1_acc", #"trend_2_acc",
        "hyper_mcc",
        "hypo_mcc",
    ]
    temp = {}
    for sub_dir in sorted(os.listdir(log_dir)):
        dir = os.path.join(log_dir, sub_dir)
        model_res = eval_dir(dir, mode="scores")
        temp[sub_dir] = {k: v for k, v in model_res.items() if k in keys}

    def find_key(model_name, temp):
        for k in temp.keys():
            if model_name in k:
                return k

    benchmark_scores = []
    model_names = ["timl", "linear", "mogru", "reselfattn", "seqconv"]
    for column_i, key in enumerate(keys):
        ax = axs[row_i][column_i]
        for benchmark in model_names:
            benchmark = find_key(benchmark, temp)
            benchmark_scores.append(eval_dir(os.path.join(log_dir, find_key(benchmark, temp)), mode="scores"))
        df = pd.DataFrame({x: y[key] for x, y in zip(model_names, benchmark_scores)})
        sns.violinplot(data=df, ax=ax)
        ax.yaxis.set_tick_params(labelsize=30)
        # ax.xaxis.set_tick_params(labelsize=20)
        ax.get_xaxis().set_visible(False)
        ax.set_title(key, fontsize=40)
    return fig


if __name__ == '__main__':
    # results table
    Path('results').mkdir(exist_ok=True)
    name = "umt1dm_cross"
    exp_dir = f"paper_log/{name}"
    exp_dir = f"storage/experiments/{name}"
    collect_as_child(experiment_dir=exp_dir, collect_keys=['anchor_patient'])
    collect_as_parent(experiment_dir=exp_dir, parent_keys=['self_train_subset_factor'])
    df = result_csv(exp_dir)
    df.to_csv(f"results/{name}.csv")

    # violin plot
    for dataset in ["ohiot1dm", "umt1dm"]:
        fig, axs = plt.subplots(2, 5, figsize=(40, 10))
        violin(f"paper_log/{dataset}_cross/self_train_subset_factor=1", fig, axs, 0)
        violin(f"paper_log/{dataset}_self", fig, axs, 1)
        fig.tight_layout(pad=1.0)
        fig.savefig(f"violin_{dataset}.pdf", format='pdf', dpi=1200)
        plt.show()
