""" Module containing utility functions for 
benchmarking MODNet against matbench datasets.

"""

MATBENCH_SEED = 18012019

from traceback import print_exc

import numpy as np
from modnet.models import MODNetModel

def plot_benchmark(model, data, val_data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    predictions = model.predict(val_data)
    errors = np.abs(predictions.values - data.df_targets.values)

    fig, axes = plt.subplots(ncols=3, facecolor="w", figsize=(12, 4))
    
    ax = axes[0]
    ax.hist(errors, bins=100)
    name = model.target_names[0].split()[:-1]
    units = model.target_names[0].split()[-1].replace(")", "").replace("(", "")
    ax.set_ylabel("Frequency")
    ax.set_xlabel(f"Absolute error")
    ax.axvline(np.mean(model.history.history["val_mae"]), 0, 1, ls='--', color='r', label="Validation MAE")
    ax.axvline(np.mean(errors), ls='--', color='b', label="Test MAE")
    ax.legend()
    
    ax = axes[1]
    ax.scatter(data.df_targets.values, predictions.values, alpha=0.25, label=f"Test MAE: {np.mean(errors):3.1f} {units}")
    ax.plot(np.linspace(*ax.get_xlim(), 2), np.linspace(*ax.get_xlim(), 2))
    sns.regplot(x=data.df_targets.values, y=predictions.values, ax=ax)
    ax.set_xlabel(f"{name} true ({units})")
    ax.set_xlabel(f"{name} pred.({units})")
    ax.legend()
    
    ax = axes[2]
    ax.plot(model.history.history["val_mae"][10:])
    if target_mae is not None:
        ax.axhline(target_mae, ls='--', color="k", label="Automatminer RF")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Validation MAE ({units})")
    plt.show(block=False)

    return predictions, errors

def matbench_kfold_splits(data):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=MATBENCH_SEED)
    kf_splits = kf.split(data.df_featurized, y=data.df_targets)
    return kf_splits

def learning_callbacks(verbose=False):
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
    return [
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=20, mode="auto", min_delta=0.0, verbose=verbose), 
        EarlyStopping(monitor="loss", min_delta=0.01, patience=300, mode="auto", baseline=None, restore_best_weights=True, verbose=verbose)
    ]

    

def matbench_benchmark(data, target, weights, fit_settings):

    all_models = [] 
    all_predictions = []
    all_errors = []
    all_maes = []
    all_targets = []

    if "n_feat" not in fit_settings or "num_neurons" not in fit_settings:
        raise RuntimeError("Need to supply n_feat or num_neurons")
    
    for ind, (train, test) in enumerate(matbench_kfold_splits(data)):
        train_data, test_data = data.split((train, test))

        model = MODNetModel(target, weights, n_feat=fit_settings["n_feat"], num_neurons=fit_settings["num_neurons"])
        model.fit(train_data, callbacks=learning_callbacks(), **fit_settings)

        try:
            predictions = model.predict(test_data).values.flatten()
            targets = test_data.df_targets[model.target_names[0]].values.flatten()
            errors = targets - predictions
            mae = np.mean(np.abs(errors))
        except Exception:
            print_exc()
            print("Something went wrong benchmarking this model.")
            predictions = None
            errors = None
            mae = None

        print(f"Model #{ind+1}: MAE = {mae}")

        all_models.append(model)
        all_predictions.append(predictions)
        all_errors.append(errors)
        all_maes.append(mae)
        all_targets.append(targets)

    print(f"Overall MAE = {np.mean(all_maes)}")

    results = {
        "models": all_models,
        "predictions": all_predictions,
        "targets": all_targets,
        "errors": all_errors,
        "maes": all_maes
    }
    
    return results
