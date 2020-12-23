""" Module containing utility functions for 
benchmarking MODNet against matbench datasets.

"""

MATBENCH_SEED = 18012019

from traceback import print_exc

import numpy as np
from modnet.models import MODNetModel
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder


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


def matbench_benchmark(data, target, weights, fit_settings,classification=False,multi=False,save_folds=False):

    all_models = [] 
    all_predictions = []
    all_errors = []
    all_scores = []
    all_targets = []

    if "n_feat" not in fit_settings or "num_neurons" not in fit_settings:
        raise RuntimeError("Need to supply n_feat or num_neurons")
    
    for ind, (train, test) in enumerate(matbench_kfold_splits(data)):
        train_data, test_data = data.split((train, test))
        
        
        if classification:
            model = MODNetModel(target, weights, n_feat=fit_settings["n_feat"],
                            num_neurons=fit_settings["num_neurons"], act = fit_settings['act'], num_classes = fit_settings["num_classes"])
        else:
            model = MODNetModel(target, weights, n_feat=fit_settings["n_feat"],
                            num_neurons=fit_settings["num_neurons"], act = fit_settings['act'])

        if fit_settings["increase_bs"]:
            model.fit(train_data, lr = fit_settings['lr'], epochs = fit_settings["epochs"], batch_size=fit_settings["batch_size"], loss = 'mse')
            model.fit(train_data, lr = fit_settings['lr']/7, epochs = fit_settings["epochs"]//2, batch_size=fit_settings["batch_size"]*2, loss = fit_settings['loss'])
        else:
            model.fit(train_data, callbacks=learning_callbacks(), **fit_settings)

        try:
            if classification:
                predictions = model.predict(test_data,return_prob=True)
            else:
                predictions = model.predict(test_data)
            targets = test_data.df_targets
            
            if classification:
                y_true = OneHotEncoder().fit_transform(targets.values).toarray()
                score = roc_auc_score(y_true,predictions.values)
                pred_bool = model.predict(test_data,return_prob=False)
                errors = targets-pred_bool
                print(f"Model #{ind+1}: ROC_AUC = {score}")
            elif multi:
                errors = targets - predictions
                score = np.mean(np.abs(errors.values),axis=0)
                print(f"Model #{ind+1}: MAE = {score}")
            else:
                errors = targets - predictions
                score = np.mean(np.abs(errors.values))
                print(f"Model #{ind+1}: MAE = {score}")
        except Exception:
            print_exc()
            print("Something went wrong benchmarking this model.")
            predictions = None
            errors = None
            score = None
            
        if save_folds:
            opt_feat = train_data.optimal_features[:fit_settings["n_feat"]]
            df_train = train_data.df_featurized
            df_train = df_train[opt_feat]
            df_train.to_csv('folds/train_f{}.csv'.format(ind+1))
            df_test = test_data.df_featurized
            df_test = df_test[opt_feat]
            errors.columns = [x+'_error' for x in errors.columns]
            df_test = df_test.join(errors)
            df_test.to_csv('folds/test_f{}.csv'.format(ind+1))
            

        all_models.append(model)
        all_predictions.append(predictions)
        all_errors.append(errors)
        all_scores.append(score)
        all_targets.append(targets)

    results = {
        "models": all_models,
        "predictions": all_predictions,
        "targets": all_targets,
        "errors": all_errors,
        "scores": all_scores
    }
    
    return results