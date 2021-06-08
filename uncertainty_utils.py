import numpy as np
from scipy import stats
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

marker_size = 0.8
diag_color = "tab:green"

def plot_color(ax, xx, yy, good):
    from matplotlib.colors import from_levels_and_colors
    from matplotlib.collections import LineCollection
    cmap, norm = from_levels_and_colors([0.0, 0.5, 1.5], ['blue', 'black'])
    points = np.array([xx, yy]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lines = LineCollection(segments, cmap=cmap, norm=norm)
    lines.set_array(good.astype(int))
    ax.add_collection(lines)

def plot_calibration(y_pred, y_std, y_true, ax, num_bins=200):
    """
    Return lists of expected and observed proportions of points falling into
    intervals corresponding to a range of quantiles.
    """

    # Compute proportions
    exp_proportions = np.linspace(0, 1, num_bins)

    norm = stats.norm(loc=0, scale=1)
    gaussian_lower_bound = norm.ppf(0.5 - exp_proportions / 2.0)
    gaussian_upper_bound = norm.ppf(0.5 + exp_proportions / 2.0)
    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    above_lower = normalized_residuals >= gaussian_lower_bound
    below_upper = normalized_residuals <= gaussian_upper_bound

    within_quantile = above_lower * below_upper
    obs_proportions = np.sum(within_quantile, axis=0).flatten() / len(residuals)

    ax.plot([0, 1], [0, 1], "--", label="Ideal", c=diag_color)
    plot_color(ax, exp_proportions, obs_proportions,(exp_proportions-obs_proportions)>0)
    #ax.scatter(exp_proportions, obs_proportions, c = (exp_proportions-obs_proportions)>0, edgecolor=None)

    ax.fill_between(exp_proportions, exp_proportions, obs_proportions,
                    where= (exp_proportions-obs_proportions)>0, interpolate = True,
                    color="black", alpha=0.2, label="Overconfident")
    ax.fill_between(exp_proportions, exp_proportions, obs_proportions,
                    where= (exp_proportions-obs_proportions)<0, interpolate=True,
                    color="blue", alpha=0.2, label="Underconfident")

    ax.set_aspect('equal', adjustable='box')
    buff = 0
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    # Compute miscalibration area
    polygon_points = []
    for point in zip(exp_proportions, obs_proportions):
        polygon_points.append(point)
    for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
        polygon_points.append(point)
    polygon_points.append((exp_proportions[0], obs_proportions[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy  # original data
    ls = LineString(np.c_[x, y])  # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    # Annotate plot with the miscalibration area
    ax.text(
        x=0.95,
        y=0.05,
        s="Miscalibration area = %.2f" % miscalibration_area,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    ax.set_xlabel("Theoritical proportion in Gaussian interval")
    ax.set_ylabel("Observed proportion in Gaussian interval")
    ax.legend()

def plot_interval(y_pred, y_std, y_true, ax, settings,ind, num_stds_confidence_bound=2):

    # randomly select 100 samples for better visualization
    #selection = np.random.choice(np.arange(len(y_pred)),100)
    #y_pred, y_std, y_true = y_pred[selection], y_std[selection], y_true[selection]
    
    intervals = num_stds_confidence_bound * y_std

    ax.errorbar(
        y_true,
        y_pred,
        yerr = intervals,
        fmt="o",
        ls="none",
        ms=marker_size,
        linewidth=marker_size,
        c="#1f77b4",
        alpha=0.5,
        zorder=1
    )
    ax.scatter(y_true, y_pred, s=marker_size, c="tab:blue",zorder=2)

    # Determine lims
    intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
    lims_ext = [
        int(np.floor(np.min(intervals_lower_upper[0]))),
        int(np.ceil(np.max(intervals_lower_upper[1]))),
    ]
    # plot 45-degree parity line
    ax.plot(lims_ext, lims_ext, "--", linewidth=marker_size, c=diag_color,zorder=3)

    # Format
    name = settings["target_names"][ind]
    ax.set_xlim(lims_ext)
    ax.set_ylim(lims_ext)
    ax.set_xlabel(f"Target {name} ({settings.get('units','dimensionless')})")
    ax.set_ylabel(f"Predicted {name} ({settings.get('units','dimensionless')})")
    ax.set_aspect("equal", "box")

def plot_interval_ordered(y_pred, y_std, y_true, ax, settings,ind, num_stds_confidence_bound=2):
    
    intervals = num_stds_confidence_bound * y_std
    order = np.argsort(y_true.flatten())
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    xs = np.arange(len(order))

    ax.errorbar(
        xs,
        y_pred,
        yerr=intervals,
        fmt="o",
        ls="none",
        linewidth=marker_size,
        c="#1f77b4",
        alpha=0.5,
        ms = marker_size,
        zorder=1
    )
    ax.scatter(xs, y_pred, s=marker_size, c="tab:blue", label='Predictions', zorder=2)
    ax.plot(xs, y_true, "--", linewidth=marker_size, c=diag_color, label='Target', zorder=3)
    ax.set_xlim([0,len(y_pred)])

    ax.legend()

    # Determine lims
    intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
    lims_ext = [
        int(np.floor(np.min(intervals_lower_upper[0]))),
        int(np.ceil(np.max(intervals_lower_upper[1]))),
    ]

    # Format
    name = settings["target_names"][ind]
    ax.set_ylim(lims_ext)
    ax.set_xlabel(f"Index ordered by {name} ({settings.get('units','dimensionless')})")
    ax.set_ylabel(f"Predicted {name} ({settings.get('units','dimensionless')})")
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))


def plot_std(y_pred, y_std, y_true, dknn, ax, settings,ind, num_stds_confidence_bound=2):

    error = np.absolute(y_pred-y_true)
    #intervals = num_stds_confidence_bound * y_std
    ax.scatter(
        y_std,
        error,
        c="#1f77b4",
        alpha=0.5,
        s=marker_size,
    )

    ax2 = ax.twinx()
    ax2.scatter(
        y_std,
        dknn,
        c="red",
        alpha=0.5,
        s=marker_size,
    )

    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red', labelsize=7)

    # Format
    name = settings["target_names"][ind]
    ax.set_xlabel("Predicted STD")
    ax.set_ylabel(f"Absolute error {name} ({settings.get('units','dimensionless')})")
    ax2.set_ylabel("$d_{KNN}$", color="red")

    _,x1 = ax.get_xlim()
    _,y1 = ax.get_ylim()
    ax.set_xlim([0,x1])
    ax.set_ylim([0,y1])
    ax.set_aspect((x1 - 0) / (y1 - 0))

    _,x1 = ax2.get_xlim()
    _,y1 = ax2.get_ylim()
    ax2.set_xlim([0,x1])
    ax2.set_ylim([0,y1])

    ax2.set_aspect((x1-0)/(y1-0))



def plot_std_by_index(y_pred, y_std, y_true, ax, settings,ind, num_stds_confidence_bound=2):

    error = np.absolute(y_pred-y_true)
    order = np.argsort(error.flatten())
    y_pred, y_std, y_true, error = y_pred[order], y_std[order], y_true[order], error[order]
    intervals = num_stds_confidence_bound * y_std
    xs = np.arange(len(order))

    ax.errorbar(
        xs,
        error,
        yerr=intervals,
        fmt="o",
        ls="none",
        ms=marker_size,
        linewidth=marker_size,
        c="#1f77b4",
        alpha=0.5,
    )
    ax.plot(xs, error, "o", ms=marker_size, c="#1f77b4")
    #ax.plot(xs, y_true, "--", linewidth=2.0, c=diag_color, label='Target')

    # Determine lims
    intervals_lower_upper = [error - intervals, error + intervals]
    lims_ext = [
        int(np.floor(np.min(intervals_lower_upper[0]))),
        int(np.ceil(np.max(intervals_lower_upper[1]))),
    ]

    # Format
    name = settings["target_names"][ind]
    #ax.set_xlim(lims_ext)
    ax.set_ylim(lims_ext)
    ax.set_xlabel(f"Index ordered by error")
    ax.set_ylabel(f"Error {name} ({settings.get('units','dimensionless')})")
    ax.set_xlim([0,len(y_pred)])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))


def plot_ordered_mae(y_pred, y_std, y_true, dknn, ax, settings,ind):

    error = np.absolute(y_pred-y_true)
    order = np.argsort(y_std.flatten())[::-1]
    std_error = error[order]

    perfect_error = np.sort(error)[::-1]


    np.random.seed(0)
    random_error = np.zeros(shape=(1000,len(error)))
    for i in range(len(random_error)):
        random_error[i,:] = np.random.permutation(error)
    random_mean = np.zeros(len(error))
    random_std = np.zeros(len(error))

    order = np.argsort(dknn.flatten())[::-1]
    dknn_error = error[order]

    for i in range(len(error)):
        perfect_error[i] = perfect_error[i:].mean()
        std_error[i] = std_error[i:].mean()
        dknn_error[i] = dknn_error[i:].mean()
        random_mean[i] = random_error[:,i:].mean()
        random_std[i] = random_error[:, i:].mean(axis=1).std()

    percintile = np.linspace(0,100,len(error))
    ax.plot(percintile, perfect_error, c='tab:green', label='Error ranked')
    ax.plot(percintile, random_mean, c='tab:red', label='Randomly ranked')
    ax.fill_between(percintile, random_mean-random_std, random_mean+random_std, color='tab:red',alpha=0.5)
    ax.plot(percintile, std_error, c='tab:blue', label='Std ranked')
    ax.plot(percintile, dknn_error, c='tab:orange', label='dKNN ranked')

    x0,x1 = 0,100
    y0,y1 = 0,random_mean[0]*1.2
    ax.set_xlim((x0, x1))
    ax.set_ylim((y0,y1))
    ax.set_aspect((x1-x0)/(y1-y0))
    ax.set_xticks([0,20,40,60,80,100])
    #ax.legend(loc='upper left')
    name = settings["target_names"][ind]
    ax.set_xlabel("Confidence percentile")
    ax.set_ylabel(f"MAE {name} ({settings.get('units','dimensionless')})")


if __name__ == "__main__":
        # for testing purposes
        import matplotlib
        import matplotlib.pyplot as plt
        DARK2_COLOURS = plt.cm.get_cmap("Dark2").colors
        matplotlib.use("pdf")
        HEIGHT = 2.5
        matplotlib.rcParams["font.size"] = 8
        matplotlib.rcParams["xtick.labelsize"] = 8
        matplotlib.rcParams["ytick.labelsize"] = 8

        print('plotting')
        fig, axs = plt.subplots(2,3,figsize=(3 * 1.25 * HEIGHT + 0.5, 2 * HEIGHT * 1.25 + 0.5))
        axs = axs.flatten()
        y = np.arange(300)
        preds = y + (np.random.rand(len(y))-0.5)*30
        unc = np.absolute(preds-y)/3 + np.random.rand(len(y))*3
        unc[unc.argsort()[-100:]] = unc[unc.argsort()[-100:]]*10
        dknn = np.absolute(preds-y)/2 + np.random.rand(len(y))*3

        settings = {'target_names':['eform'], 'units':'eV'}

        plot_calibration(preds, unc, y, axs[0])
        plot_interval(preds, unc, y, axs[1], settings, 0)
        plot_interval_ordered(preds, unc, y, axs[2], settings, 0)
        plot_std(preds, unc, y, axs[3], settings, 0)
        plot_std_by_index(preds, unc, y, axs[4], settings, 0)
        plot_ordered_mae(preds, unc, y, dknn, axs[5], settings, 0)

        fig.tight_layout()
        fig.suptitle('Dummy test plot')
        fig.savefig('test_fig.pdf')