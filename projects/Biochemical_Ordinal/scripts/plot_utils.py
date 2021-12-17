import scipy as sp
import matplotlib.pyplot as plt

import warnings
# scipy gives a warning with the input arrays to correlation are constant
warnings.filterwarnings("ignore", message = ".*An input array is constant.*")


def annotate_ax_with_correlations(x, y, ax=None, *args, **kwargs):
    pearson_r = sp.stats.pearsonr(x, y)
    spearman_r = sp.stats.spearmanr(x, y)
    if ax is None:
        ax = plt.gca()
    ax.text(0.05, 0.8, f"{pearson_r[0]:.2f} : pearson_r\n"
                   f"{spearman_r[0]:.2f}: spearman_r", 
        horizontalalignment='left',
        bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5),
        transform=ax.transAxes)
