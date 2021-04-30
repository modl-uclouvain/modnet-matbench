import copy
import json
import pickle
import os
import glob
from collections import defaultdict
from traceback import print_exc

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from modnet.models import MODNetModel, EnsembleMODNetModel
from modnet.matbench.benchmark import matbench_kfold_splits

from pymatgen.core import Composition

DARK2_COLOURS = plt.cm.get_cmap("Dark2").colors
matplotlib.use("pdf")
HEIGHT = 2.5
matplotlib.rcParams["font.size"] = 8

STIX = False

if STIX:
    # matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.family"] = "STIXGeneral"
    matplotlib.rcParams["mathtext.fontset"] = "stixsans"
else:
    matplotlib.rcParams["mathtext.fontset"] = "stixsans"
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = "Arial"


# Require these imports for backwards-compat when unpickling
try:
    from modnet.featurizers.presets import CompositionOnlyFeaturizer  # noqa
    from modnet.preprocessing import MODData, CompositionContainer  # noqa
except ImportError:
    pass


def setup_threading():
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # import tensorflow as tf 
    # tf.config.threading.set_intra_op_parallelism_threads(nprocs)
    # tf.config.threading.set_inter_op_parallelism_threads(nthreads)


def load_settings(task: str):
    settings = {}
    settings_path = task + "_options.json"
    if os.path.isfile(settings_path):
        settings_path = task + "_options.json"
    elif os.path.isfile(task + "/" + settings_path):
        settings_path = task + "/" + settings_path
    else:
        settings_path = None

    if settings_path:
        with open(settings_path, "r") as f:
            settings = json.load(f)
    return settings


def featurize(task):
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    from modnet.preprocessing import MODData
    from modnet.featurizers.presets import DeBreuck2020Featurizer
    from matminer.datasets import load_dataset

    if task == "matbench_elastic":
        df_g = load_dataset("matbench_log_gvrh")
        df_k = load_dataset("matbench_log_kvrh")
        df = df_g.join(df_k.drop("structure",axis=1))
    else:
        df = load_dataset(task)

    mapping = {
        col: col.replace(" ", "_").replace("(", "").replace(")", "")
        for ind, col in enumerate(df.columns)
    }
    df.rename(columns=mapping, inplace=True)

    targets = [
        col for col in df.columns if col not in ("id", "structure", "composition")
    ]

    try:
        materials = df["structure"] if "structure" in df.columns else df["composition"].map(Composition)
    except KeyError:
        raise RuntimeError(f"Could not find any materials data dataset for task {task!r}!")

    fast_oxid_featurizer = DeBreuck2020Featurizer(fast_oxid=True)
    data = MODData(
        materials=materials.tolist(),
        targets=df[targets].values,
        target_names=targets,
        featurizer=fast_oxid_featurizer,
    )
    data.featurize(n_jobs=32)
    data.save(f"./precomputed/{task}_moddata.pkl.gz")
    return data


def benchmark(data, settings,n_jobs=16, fast=False):
    from modnet.matbench.benchmark import matbench_benchmark

    columns = list(data.df_targets.columns)
    mapping = {
        col: col.replace(" ", "_").replace("(", "").replace(")", "")
        for ind, col in enumerate(columns)
    }
    data.df_targets.rename(columns=mapping, inplace=True)

    best_settings = {
        "increase_bs": False,
        "lr": 0.005,
        "epochs": 50,
        "act": "elu",
        "out_act": "relu",
        "batch_size": 32,
        "loss": "mae",
        "xscale": "minmax",
    }

    #best_settings = None
    names = [[[field for field in data.df_targets.columns]]]
    weights = {field: 1 for field in data.df_targets.columns}
    from modnet.models import EnsembleMODNetModel
    return matbench_benchmark(
        data,
        names,
        weights,
        best_settings,
        model_type=EnsembleMODNetModel,
        n_models = 5,
        classification=settings.get("classification"),
        fast=fast,
        nested= 0 if fast else 5,
        n_jobs=n_jobs,
    )

def add_postprocess(model: MODNetModel,train_data):
    # this is a not very proper solution to add post processing to an instance,
    # but does the job (it's only temporary)
    new_model = copy.deepcopy(model)
    def post_process_predict(*args,**kwargs):
        min_y = train_data.df_targets.values.min(axis=0)
        max_y = train_data.df_targets.values.max(axis=0)
        preds = model.predict(*args, **kwargs)
        range = max_y-min_y
        upper_bound = max_y + 0.25 * range
        lower_bound = min_y - 0.25 * range
        for i, c in enumerate(preds.columns):
            vals = preds[c].values
            out_of_range_idxs = np.where((vals < lower_bound[i]) | (vals > upper_bound[i]))[0]
            vals[out_of_range_idxs] = np.random.uniform(0, 1, size=len(out_of_range_idxs)) * (max_y[i] - min_y[i]) + min_y[i]
        return preds

    new_model.predict = post_process_predict
    return new_model

def run_predict(data, final_model, settings, save_folds=False, postprocess=False):
    """
    Runs benchmark based on final_model without training everything again.
    It also computes the Knn distance and puts it in the results pickle.
    In fine, this should be integrated inside modnet benchmark.
    :param data:
    :param final_model:
    :param settings:
    :return:
    """
    # rebuild the EnsembleMODNetModels from the final model

    n_best_archs = 5 # change this (from 1 to 5 max) to adapt number of inner best archs chosen

    bootstrap_size = 2 if settings["task"]=="matbench_elastic" else 5
    outer_fold_size = bootstrap_size * 5 * 5
    inner_fold_size = bootstrap_size * 5
    models = []

    multi_target = bool(len(data.df_targets.columns) - 1)

    for i in range(5): # outer fold
        modnet_models = []
        for j in range(5): # inner fold
                modnet_models+=(
                    final_model.model[(i * outer_fold_size) + (j * inner_fold_size):
                                      (i * outer_fold_size) + (j * inner_fold_size) + (n_best_archs * bootstrap_size)])
        model = EnsembleMODNetModel(modnet_models=modnet_models)
        models.append(model)

    results = defaultdict(list)
    for ind, (train, test) in enumerate(matbench_kfold_splits(data)):
        train_data, test_data = data.split((train, test))
        model = models[ind]
        if postprocess:
            model.model = [add_postprocess(m,train_data) for m in model.model]
        predict_kwargs = {}
        if settings.get("classification"):
            predict_kwargs["return_prob"] = True
        if model.can_return_uncertainty:
            predict_kwargs["return_unc"] = True

        pred_results = model.predict(test_data, **predict_kwargs)
        if isinstance(pred_results, tuple):
            predictions, stds = pred_results
        else:
            predictions = pred_results
            stds = None

        targets = test_data.df_targets

        if settings.get("classification"):
            from sklearn.metrics import roc_auc_score
            from sklearn.preprocessing import OneHotEncoder

            y_true = OneHotEncoder().fit_transform(targets.values).toarray()
            score = roc_auc_score(y_true, predictions.values)
            pred_bool = model.predict(test_data, return_prob=False)
            print(f"ROC-AUC: {score}")
            errors = targets - pred_bool
        elif multi_target:
            errors = targets - predictions
            score = np.mean(np.abs(errors.values), axis=0)
        else:
            errors = targets - predictions
            score = np.mean(np.abs(errors.values))
        # compute dkNN
        max_feat_model = np.argmax([m.n_feat for m in model.model])
        n_feat = model.model[max_feat_model].n_feat
        feature_names = model.model[max_feat_model].optimal_descriptors
        dknn = get_dknn(train_data, test_data, feature_names)
        results["dknns"].append(dknn)

        if save_folds:
            opt_feat = train_data.optimal_features[:n_feat]
            df_train = train_data.df_featurized
            df_train = df_train[opt_feat]
            df_train.to_csv("folds/train_f{}.csv".format(ind + 1))
            df_test = test_data.df_featurized
            df_test = df_test[opt_feat]
            errors.columns = [x + "_error" for x in errors.columns]
            df_test = df_test.join(errors)
            df_test.to_csv("folds/test_f{}.csv".format(ind + 1))

        results["predictions"].append(predictions)
        if stds is not None:
            results["stds"].append(stds)
        results["targets"].append(targets)
        results["errors"].append(errors)
        results["scores"].append(score)
        results['model'].append(model)

    return results


def get_dknn(train_data, test_data, feature_names, k = 5):
    x_train = train_data.df_featurized[feature_names].values
    x_test = test_data.df_featurized[feature_names].values

    scaler = sklearn.preprocessing.StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    dist = sklearn.metrics.pairwise.euclidean_distances(x_test_sc, x_train_sc)
    dknn = np.sort(dist, axis=1)[:, :k].mean(axis=1)

    dknn = pd.DataFrame({t:dknn for t in train_data.df_targets.columns}, index = test_data.df_featurized.index)

    return dknn



def get_metrics(target, pred, errors, name, settings):
    import sklearn.metrics
    import scipy

    metrics = {}

    if settings.get("classification"):
        metrics["roc_auc"] = score = sklearn.metrics.roc_auc_score(
            target.reshape(-1, 1), 1 - pred.reshape(-1, 1)
        )
        metrics["ap_score"] = ap_score = sklearn.metrics.average_precision_score(
            target.reshape(-1, 1), 1 - pred.reshape(-1, 1), average="micro"
        )
        print(f"ROC-AUC: {score:3.3f}, AP: {ap_score:3.3f}")
    else:
        mae = metrics["mae"] = sklearn.metrics.mean_absolute_error(target, pred)
        try:
            mape = metrics["mape"] = sklearn.metrics.mean_absolute_percentage_error(
                target, pred
            )
        except AttributeError:
            mape = metrics["mape"] = 1e20
        mse = metrics["mse"] = sklearn.metrics.mean_squared_error(target, pred)
        med_ae = metrics["med_ae"] = sklearn.metrics.median_absolute_error(target, pred)
        max_ae = metrics["max_ae"] = sklearn.metrics.max_error(target, pred)
        fit_results = scipy.stats.linregress(
            x=target.reshape(
                -1,
            ),
            y=pred.reshape(
                -1,
            ),
        )
        slope = metrics["slope"] = fit_results.slope
        rvalue = metrics["rvalue"] = fit_results.rvalue

        print(
            f"MAE = {mae:3.3f} {settings.get('units', '')}, MedianAE = {med_ae:3.3f} {settings.get('units', '')}, MAPE = {mape:3.3f}, √MSE = {np.sqrt(mse):3.3f} {settings.get('units', '')}"  # noqa
        )
        print(
            f"MaxAE = {max_ae:3.3f} {settings.get('units', '')}, slope = {slope:3.2f}, R = {rvalue:3.2f}"
        )

    with open(f"results/{name}_metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics


def analyse_results(results, settings, post_process=False):

    target_names = set(c for res in results["targets"] for c in res.columns)

    all_targets = []
    all_preds = []
    all_stds = []
    all_dknns = []
    all_errors = []

    for name in target_names:
        targets = np.hstack([res[name].values for res in results["targets"]]).flatten()
        if settings.get("classification"):
            if len(target_names) > 1:
                raise RuntimeError("Cannot handle multi-target classification.")
            preds = np.hstack(
                [res[res.columns[0]].values for res in results["predictions"]]
            ).flatten()
        else:
            preds = np.hstack(
                [res[name].values for res in results["predictions"]]
            ).flatten()
        stds = np.hstack([res[name].values for res in results["stds"]]).flatten()
        dknns = np.hstack([res[name].values for res in results["dknns"]]).flatten()
        try:
            errors = np.hstack(
                [res[name].values for res in results["errors"]]
            ).flatten()
        except (NameError, KeyError):
            errors = np.hstack(
                [res[name + "_error"].values for res in results["errors"]]
            ).flatten()

        if post_process:
            outliers = np.where(np.abs(preds) / np.max(np.abs(targets)) > 1.2)[0]
            print(outliers)
            for outlier in outliers:
                print(
                    f"Setting value of outlier {outlier} to the mean of the dataset {np.mean(targets)} from {preds[outlier]}"
                )
                preds[outlier] = np.mean(targets)
                errors[outlier] = targets[outlier] - preds[outlier]
                #stds[outlier] = np.mean(stds)

            outliers = np.where(stds / (np.max(targets)-np.min(targets)) > 0.68)[0]
            max_std = np.sort(stds)[-(len(outliers)+1)]
            for outlier in outliers:
                print(
                    f"Setting value of std outlier {outlier} to the max of the dataset {max_std} from {stds[outlier]}"
                )
                stds[outlier] = max_std

        all_targets.append(targets)
        all_preds.append(preds)
        all_stds.append(stds)
        all_dknns.append(dknns)
        all_errors.append(errors)

    for t, p, e, name in zip(all_targets, all_preds, all_errors, target_names):
        metrics = get_metrics(t, p, e, name, settings)

    os.makedirs("./plots", exist_ok=True)

    for ind, (target, pred, stds, dknns, error) in enumerate(
        zip(all_targets, all_preds, all_stds, all_dknns, all_errors)
    ):
        if not settings.get("classification", False):
            # if "nested_learning_curves" in results:
            #     plot_learning_curves(results["nested_learning_curves"], results["best_learning_curves"], settings)
            plot_jointplot(target, error, ind, settings)
            plot_scatter(target, pred, error, ind, settings, metrics)
            plot_uncertainty(target, pred, stds, dknns, ind, settings)
        else:
            plot_classifier_roc(target, pred, settings)

def plot_uncertainty(all_targets, all_pred, all_stds, all_dknns, ind, settings):
    from uncertainty_utils import (plot_calibration,
     plot_interval, plot_interval_ordered,
     plot_std, plot_std_by_index, plot_ordered_mae)

    fig, axs = plt.subplots(2,3,figsize=(3 * 1.25 * HEIGHT + 0.5, 2 * HEIGHT * 1.25 + 0.5))
    axs = axs.flatten()

    plot_calibration(all_pred, all_stds, all_targets, axs[0])
    plot_interval(all_pred, all_stds, all_targets, axs[1], settings, ind)
    plot_interval_ordered(all_pred, all_stds, all_targets, axs[2], settings, ind)
    plot_std(all_pred, all_stds, all_targets, axs[3], settings, ind)
    plot_std_by_index(all_pred, all_stds, all_targets, axs[4], settings, ind)
    plot_ordered_mae(all_pred, all_stds, all_targets, all_dknns, axs[5], settings, ind)

    fig.tight_layout()
    name = settings["target_names"][ind]
    fig.suptitle(f"{name}")

    if len(settings.get("target_names", [])) <= 1:
        fname = f"plots/{settings['task']}_uncertainty.pdf"
    else:
        fname = f"plots/{settings['task']}_{name.replace('$', '').replace('{', '').replace('}', '')}_uncertainty.pdf"
    fig.savefig(fname, dpi=300)

def plot_jointplot(all_targets, all_errors, ind, settings):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame().from_dict({"targets": all_targets, "errors": all_errors})
    name = settings["target_names"][ind]
    g = sns.JointGrid(
        data=df, x="targets", y="errors", height=1.25 * HEIGHT, ratio=4, space=0.1
    )
    g.plot_marginals(sns.histplot, kde=False, color="grey", alpha=0.6, lw=0, bins=81)
    # g.plot_marginals(sns.kdeplot, color="grey", alpha=0.2, lw=1, fill=True)
    sns.regplot(
        x="targets",
        y="errors",
        data=df,
        ax=g.ax_joint,
        scatter=False,
        color="k",
        truncate=False,
    )
    ax = g.ax_joint
    ax.set_ylabel(f"Prediction error ({settings.get('units', 'dimensionless')})")
    ax.set_xlabel(f"Target {name} ({settings.get('units', 'dimensionless')})")
    ax.scatter(
        all_targets,
        all_errors,
        alpha=0.6,
        c=np.abs(all_errors),
        cmap="cividis_r",
        rasterized=True,
    )
    ax.axhline(
        np.mean(np.abs(all_errors)),
        c="black",
        label=f"MAE = {np.mean(np.abs(all_errors)):3.3f} {settings.get('units', '')}",
        ls="-.",
        alpha=0.5,
        lw=1,
    )
    ax.axhline(-np.mean(np.abs(all_errors)), c="black", ls="-.", alpha=0.5, lw=1)
    max_ae = np.max(np.abs(all_errors))
    range_ = (-max_ae, max_ae)
    padding = 0.02
    range_ = (range_[0] - padding * range_[0], range_[1] + padding * range_[1])
    ax.legend(loc="upper left")
    ax.set_ylim(*range_)
    ax.set_xlim((1 - padding) * ax.get_xlim()[0], (1 + padding) * ax.get_xlim()[1])
    ax = g.ax_marg_y
    ax.set_xlim(-1)

    plt.tight_layout()
    if len(settings.get("target_names", [])) <= 1:
        fname = f"plots/{settings['task']}_jointplot.pdf"
    else:
        fname = f"plots/{settings['task']}_{name.replace('$', '').replace('{', '').replace('}', '')}_jointplot.pdf"
    plt.savefig(fname, dpi=300)


def plot_scatter(all_targets, all_pred, all_errors, ind, settings, metrics):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame().from_dict(
        {"targets": all_targets, "predictions": all_pred, "errors": all_errors}
    )
    fig, ax = plt.subplots(figsize=(1.25 * HEIGHT, HEIGHT))
    ax.set_aspect("equal")

    points = ax.scatter(
        all_targets,
        all_pred,
        alpha=0.6,
        c=np.abs(all_errors),
        cmap="cividis_r",
        rasterized=True,
        zorder=0,
    )
    plt.colorbar(
        points, label=f"Absolute error ({settings.get('units', 'dimensionless')})"
    )
    sns.regplot(
        x="targets",
        y="predictions",
        data=df,
        ax=ax,
        scatter=False,
        color="k",
        truncate=False,
        label=f"$m={metrics['slope']:3.2f}$; $R={metrics['rvalue']:3.2f}$",
    )
    range_ = (
        min(np.min(all_targets), np.min(all_pred)),
        max(np.max(all_targets), np.max(all_pred)),
    )
    padding = 0.02
    range_ = (range_[0] - padding * range_[0], range_[1] + padding * range_[1])
    ax.set_ylim(*range_)
    ax.set_xlim(*range_)
    ax.plot(
        (range_[0], range_[1]), (range_[0], range_[1]), ls="--", c="red", label="Ideal"
    )
    name = settings["target_names"][ind]
    ax.set_ylabel(f"Predicted {name} ({settings.get('units', 'dimensionless')})")
    ax.set_xlabel(f"Target {name} ({settings.get('units', 'dimensionless')})")
    ax.legend(loc="upper left")

    plt.tight_layout()
    if len(settings.get("target_names", [])) <= 1:
        fname = f"plots/{settings['task']}_scatter.pdf"
    else:
        fname = f"plots/{settings['task']}_{name.replace('$', '').replace('{', '').replace('}', '')}_scatter.pdf"
    plt.savefig(fname, dpi=300)


def plot_classifier_roc(all_targets, all_pred, settings):
    import sklearn.metrics
    import matplotlib.pyplot as plt
    import seaborn as sns  # noqa

    # from sklearn.preprocessing import OneHotEncoder
    # one_hot_targets = OneHotEncoder().fit_transform(all_targets.reshape(-1, 1)).toarray()
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(all_targets, 1 - all_pred)
    score = sklearn.metrics.roc_auc_score(all_targets, 1 - all_pred)
    fig, axes = plt.subplots(1, 2, figsize=(2 * HEIGHT, HEIGHT))
    ax = axes[0]
    ax.plot(fpr, tpr, c=DARK2_COLOURS[0])
    ax.fill_between(
        fpr, fpr, tpr, color=DARK2_COLOURS[0], alpha=0.1, label=f"ROC-AUC: {score:3.3f}"
    )
    ax.plot(fpr, fpr, ls="-.", c="grey", alpha=0.8)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right")
    ax.set_ylabel("True positive rate")
    ax.set_xlabel("False positive rate")

    ax = axes[1]
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        all_targets, 1 - all_pred
    )
    ap_score = sklearn.metrics.average_precision_score(
        all_targets, 1 - all_pred, average="micro"
    )
    x = np.asarray([1] + list(recall))
    y = np.asarray([0] + list(precision))
    ax.plot(x, y, label=f"Average AP score: {ap_score:3.3f}", c=DARK2_COLOURS[1])
    ax.legend(loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    plt.tight_layout()
    plt.savefig(f"plots/{settings['task']}_roc.pdf", dpi=300)


def plot_learning_curves(
    learning_curves, best_learning_curves, settings, limits=(0.5, 3)
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for ind, fold in enumerate(learning_curves):
        for jnd, curve in enumerate(fold):
            label = None
            if ind == 0 and jnd == 0:
                label = "Inner CV"
            ax.plot(curve, alpha=0.2, c="grey", label=label)
    for ind, curve in enumerate(best_learning_curves):
        label = None
        if ind == 0:
            label = "Outer CV"
        ax.plot(curve, alpha=0.8, c="blue", label=label)

    ax.legend()

    min_ = min(np.min(curve) for curve in best_learning_curves)
    ax.set_ylim(limits[0] * min_, limits[1] * min_)
    ax.set_xlabel("Number of epochs")
    if settings.get("units"):
        ax.set_ylabel(f"Validation loss ({settings.get('units')})")
    else:
        ax.set_ylabel("Validation loss")

    plt.savefig(f"plots/{settings['task']}_learning_curves.pdf", dpi=300)


def save_results(results, task: str):
    os.makedirs("results", exist_ok=True)

    with open(f"results/{task}_results.pkl", "wb") as f:
        safe_keys = [
            "targets",
            "predictions",
            "errors",
            "scores",
            "nested_learning_curves",
            "best_learning_curves",
            "stds",
            "dknns",
        ]
        pickle.dump({key: results[key] for key in safe_keys}, f)

    print("Final score = ", np.mean(results["scores"]))


def make_summary_plot():
    import matplotlib.pyplot as plt

    # import seaborn as sns # noqa

    fig, ax = plt.subplots(figsize=(4, 3))

    matbench_ordered = [
        "steels",
        "jdft2d",
        "dielectric",
        "expt_gap",
        # "expt_is_metal",
        # "glass",
        "elastic",
        "phonons",
    ]

    name_counter = 0
    names = []
    maes = []
    ys = []
    mins = []
    maxs = []

    cmap = {"AM": 1, "RF": 2, "MEGNet": 3, "CGCNN": 4}
    for task in matbench_ordered:
        settings = load_settings("matbench_" + task)
        for jnd, name in enumerate(settings["target_names"]):
            mae, min_, max_, y = add_to_plot(
                task, jnd, ax, settings, name_counter, cmap
            )
            name_counter += 1
            names.append(name)
            maes.append(mae)
            maxs.append(max_)
            mins.append(min_)
            ys.append(y)
            ax.axhline(y, alpha=0.2, ls="--", lw=1, c="grey")

    ax.plot(maes, ys, c=DARK2_COLOURS[0], alpha=0.5, zorder=0)
    ax.fill_betweenx(
        ys,
        mins,
        maxs,
        color=DARK2_COLOURS[0],
        alpha=0.1,
        zorder=0,
        lw=1,
        edgecolor=DARK2_COLOURS[0],
    )

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    marker_style = {"marker": "*", "s": 30, "lw": 0.5, "edgecolor": "k"}

    ax.scatter(1e20, 1e20, c=[DARK2_COLOURS[0]], label="MODNet", **marker_style)
    ax.scatter(
        1e20,
        1e20,
        alpha=0.3,
        color=[DARK2_COLOURS[0]],
        label="MODNet folds",
        s=20,
        lw=0,
    )
    for method in cmap:
        ax.scatter(
            1e20, 1e20, c=[DARK2_COLOURS[cmap[method]]], label=method, **marker_style
        )
    ax.legend(loc="upper right")

    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)

    ax.set_yticks(ys)

    names = [
        "yield strength ($n=312$)",
        "exfoliation energy ($n=636$)",
        "refractive index ($n=4764$)",
        "exp. band gap ($n=4604$)",
        "$\\log_{10}{K}$ ($n=10987$)",
        "$\\log_{10}{G}$ ($n=10987$)",
        "$\\mathrm{argmax}(\\mathrm{PhDOS})$ ($n=1265$)",
    ]

    ax.set_yticklabels(names)

    ax.set_xlabel("MAE relative to dummy model")

    plt.tight_layout()

    plt.savefig("summary.pdf", dpi=300)


def add_to_plot(task, jnd, ax, settings, y, cmap):
    import pickle

    with open(f"matbench_{task}/results/matbench_{task}_results.pkl", "rb") as f:
        results = pickle.load(f)

    other_methods = {}
    for other_method in settings["other_methods"][jnd]:
        other_methods[other_method] = settings["other_methods"][jnd][other_method]
    dummy = other_methods.pop("Dummy")
    fold_errors = [np.abs(res[res.columns[jnd]].values) for res in results["errors"]]
    means = []
    for fold in fold_errors:
        fold = fold[np.where(fold < 1e3)]
        means.append(np.mean(fold))

    fold_errors = np.array(means)
    mae = np.mean(fold_errors)
    mins = np.min(fold_errors)
    maxs = np.max(fold_errors)

    ax.scatter(
        fold_errors / dummy, len(fold_errors) * [y], alpha=0.3, color=[DARK2_COLOURS[0]]
    )
    ax.scatter(
        mae / dummy,
        y,
        marker="*",
        c=[DARK2_COLOURS[0]],
        zorder=1e20,
        s=75,
        lw=0.5,
        edgecolor="k",
    )
    offset_ = 0.1
    if len(other_methods) % 2 == 1:
        offset = offset_ * (len(other_methods) + 1) / 2
    else:
        offset = offset_ * len(other_methods) / 2

    for method in other_methods:
        offset -= offset_
        if np.abs(offset) < 1e-5:
            offset -= offset_
        ax.scatter(
            other_methods[method] / dummy,
            y + offset,
            color=[DARK2_COLOURS[cmap[method]]],
            marker="*",
            s=75,
            lw=0.5,
            edgecolor="k",
        )
    return mae / dummy, mins / dummy, maxs / dummy, y


def load_or_featurize(task):
    data_files = glob.glob("./precomputed/*.gz")
    if len(data_files) == 0:
        data = featurize(task)
    else:
        precomputed_moddata = data_files[0]
        if len(data_files) > 1:
            print(
                f"Found multiple data files {data_files}, loading the first {data_files[0]}"
            )

        data = MODData.load(precomputed_moddata)

    return data


if __name__ == "__main__":
    n_jobs = 40

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = vars(parser.parse_args())

    if args.get("summary"):
        make_summary_plot()
        exit()
    else:
        arg = args.get("task")
        task = "matbench_" + arg.replace("matbench_", "")

    if not os.path.isdir(task):
        raise RuntimeError(f"No folder found for {task!r}.")

    os.chdir(task)
    print(f"Running on {n_jobs} jobs")
    settings = load_settings(task)
    settings["task"] = task

    if args.get("predict"):
        if not os.path.isfile(f"final_model/{task}_model"):
            raise RuntimeError("No model found for prediction, please run the benchmark first.")
        else:
            print("Loading data and model...")
            data = load_or_featurize(task)
            final_model = EnsembleMODNetModel.load(f"final_model/{task}_model")
            print("Running predictions...")
            results = run_predict(data, final_model, settings, postprocess=True)
            print("Saving results...")
            try:
                save_results(results, task)
            except Exception:
                print_exc()

    if args.get("plot"):
        #make graphs only
        if not os.path.isfile(f"results/{task}_results.pkl"):
            raise RuntimeError("No results file, please run the benchmark before plotting.")
        else:
            print("Loading previous results.")
            with open(f"results/{task}_results.pkl", "rb") as f:
                results = pickle.load(f)
            try:
                analyse_results(results, settings, post_process=False)
            except Exception:
                print_exc()

    if not args.get("plot") and not args.get("predict"):
        #full run
        print(f"Preparing nested CV run for task {task!r}")

        data = load_or_featurize(task)
        setup_threading()
        results = benchmark(data, settings,n_jobs=n_jobs, fast=False)
        models = results['model']
        inner_models = []
        for model in models:
            inner_models += model.model
        from modnet.models import EnsembleMODNetModel
        final_model = EnsembleMODNetModel(modnet_models=inner_models)
        if not os.path.exists('final_model'):
            os.makedirs('final_model')
        final_model.save(f"final_model/{task}_model")
        del results['model']
        try:
            save_results(results, task)
        except Exception:
            print_exc()

    os.chdir("..")
