""" Module containing utility functions for 
benchmarking MODNet against matbench datasets.

"""

MATBENCH_SEED = 18012019

def plot_benchmark(model, data, target_mae=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    predictions = model.predict(data)
    errors = np.abs(predictions.values - data.df_targets.values)
    fig, axes = plt.subplots(ncols=3, facecolor="w", figsize=(12, 4))
    
    ax = axes[0]
    ax.hist(errors, bins=100)
    name = model.target_names[0].split()[:-1]
    units = model.target_names[0].split()[-1].replace(")", "").replace("(", "")
    ax.set_ylabel("Frequency")
    ax.set_xlabel(f"Absolute error ({units})")
    if target_mae is not None:
        ax.axvline(target_mae, ls='--', color='k', label="Automatminer RF")
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

def matbench_kfold_splits(data):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=MATBENCH_SEED)
    kf_splits = kf.split(data.df_featurized, y=data.df_targets)
    return kf_splits
