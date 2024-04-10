import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
import math

from .constants import PlottingConstants

def plot_time_series(ax, y, t=None, **kwargs):
    """Generic function for plotting time series.

    Plots the inputted time series signal(s) `y` on the inputted Matplotlib
    axis `ax`. This function modifies `ax` directly and does not return a new
    axis.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Matplotlib axis object on which to plot
    y : ndarray
        Numpy array holding the time series signal(s) to plot; `y` can be
        either one-dimensional (to plot a single time series) or
        two-dimensional (to plot multiple time series on the same axes). If `y`
        is two-dimensional, each time series should be a column in `y`.
    t : ndarray, optional
        1D Numpy array holding the time (or other independent variable) values
        on which `y` is defined. If provided, `t` must be a 1N Numpy array with
        the same number of points as the time series in `y`. If not provided,
        this function will use index values of `y` as the independent variable,
        i.e. `t = [0, 1, 2, ..., y.shape[0]]`.

    Keyword Arguments
    -----------------
    xlabel : str
        Label for x axis
    ylabel : str
        Label for y axis
    title : str
        Axis title
    color : str or list
        Color for the lines of plotted time series signal(s). If a list,
        `color` should contain one exactly string color for every time series
        in `y`.
    linewidth : str or list
        Width for the lines of plotted time series signal(s); str/list behavior
        as for `color`.
    marker : str or list
        Marker for the lines of plotted time series signal(s); str/list behavior
        as for `color`.
    label : str or list
        Human-readable label for the plotted time series signal(s); str/list
        behavior as for `color`.

    """
    if t is None:
        t = np.arange(y)

    xlabel=kwargs.get('xlabel', PlottingConstants.TIME_SERIES_DEFAULTS['xlabel'])
    ylabel=kwargs.get('ylabel', PlottingConstants.TIME_SERIES_DEFAULTS['ylabel'])
    title=kwargs.get('title', PlottingConstants.TIME_SERIES_DEFAULTS['title'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _remove_spines(ax)
    ax.plot(t, y, color=kwargs.get('color'), marker=kwargs.get('marker', PlottingConstants.TIME_SERIES_DEFAULTS['marker']),
            linewidth=kwargs.get('linewidth'), label=kwargs.get('label'))


def plot_spm_t_statistic(ax, spm_ts, spm_ti, t=None, **kwargs):
    """Plots an SPM t-statistic and accompanying inference results.

    Plots the inputted SPM t-statistic curve `spm_ts` and a summary of
    the accompanying inference object `spm_ti` on the inputted
    Matplotlib axes object `ax`. This function modifies `ax` directly and does
    not return a new axis.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Matplotlib axis object on which to plot
    spm_ts : SpmTStatistic
        A SpmTStatistic namedtuple, as returned by `spm.get_spm_t_statistic()`,
        holding the SPM t-test statistic curve to be plotted.
    spm_ti : SpmTInference
        A SpmTInference namedtuple produced by performing inference on
        `spm_ts` using the function `spm.get_spm_t_inference()`.
    t : ndarray, optional
        1D Numpy array holding the time (or other independent variable) values
        on which the SPM t-statistic curve `spm_ts.t_statistic` is
        defined. If provided, `t` must be a 1N Numpy array with the same number
        of points as the t-statistic curve. If not provided, this function will
        use index values of `spm_ts.t_statistic` as the independent
        variable.

    Keyword Arguments
    -----------------
    xlabel : str
        Label for x axis
    ylabel : str
        Label for y axis
    title : str
        Axis title
    color : str
        Color of the plotted t-statistic curve.
    color : str
        Color of the plotted t-statistic curve.
    marker : str
        Marker for the t-statistic curve.
    cluster_fillcolor : str
        Background color of suprathreshold significance clusters.
    threshold_color : str
        Color of the significance threshold line.
    threshold_linestyle : str
        Style of the significance threshold line.

    """
    if t is None:
        t = np.arange(y)

    xlabel=kwargs.get('xlabel', PlottingConstants.SPM_STATISTIC_DEFAULTS['xlabel'])
    ylabel=kwargs.get('ylabel', PlottingConstants.SPM_STATISTIC_DEFAULTS['ylabel'])
    title=kwargs.get('title', PlottingConstants.SPM_STATISTIC_DEFAULTS['title'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _remove_spines(ax)

    spmt = spm_ts.t_statistic
    threshold = spm_ti.threshold

    # Plot t-statistic
    ax.plot(t, spmt, color=kwargs.get('color'),
            marker=kwargs.get('marker', PlottingConstants.SPM_STATISTIC_DEFAULTS['marker']),
            linewidth=kwargs.get('linewidth'), label=kwargs.get('label'))

    # Plot dashed line at y = 0
    ax.axhline(y=0, color=PlottingConstants.SPM_STATISTIC_DEFAULTS['x_axis_color'], linestyle=PlottingConstants.SPM_STATISTIC_DEFAULTS['x_axis_linestyle'])

    # Plot dashed line at SPM significance threshold
    threshold_color = kwargs.get('threshold_color', PlottingConstants.SPM_STATISTIC_DEFAULTS['threshold_color'])
    threshold_linestyle = kwargs.get('threshold_linestyle', PlottingConstants.SPM_STATISTIC_DEFAULTS['threshold_linestyle'])
    ax.axhline(y=threshold, color=threshold_color,
               linestyle=threshold_linestyle)
    ax.axhline(y=-threshold, color=threshold_color,
               linestyle=threshold_linestyle)

    # Shade between curve and threshold
    fill_color = kwargs.get('cluster_fillcolor', PlottingConstants.SPM_STATISTIC_DEFAULTS['cluster_fillcolor'])
    ax.fill_between(t, spmt, threshold, where=spmt >= threshold,
            interpolate=True, color=fill_color)
    ax.fill_between(t, spmt, -threshold, where=spmt <= -threshold,
            interpolate=True, color=fill_color)

    # Text box showing SPM cluster parameters
    best_cluster = None if len(spm_ti.clusters) == 0 else spm_ti.clusters[_choose_cluster_to_display(spm_ti.clusters)]
    ax.text(PlottingConstants.SPM_STATISTIC_DEFAULTS['textbox_x'], PlottingConstants.SPM_STATISTIC_DEFAULTS['textbox_y'],
            _get_spm_axis_text(spm_ti.alpha, spm_ti.threshold, best_cluster),
            va='top', ha='left',
            transform=ax.transAxes,
            bbox=dict(facecolor=PlottingConstants.SPM_STATISTIC_DEFAULTS['textbox_facecolor'],
                      edgecolor=PlottingConstants.SPM_STATISTIC_DEFAULTS['textbox_edgecolor'],
                      boxstyle=PlottingConstants.SPM_STATISTIC_DEFAULTS['textbox_style']))


def plot_spm_input_data(ax, group1, group2, t=None, **kwargs):
    """Plots the data used to compute an SPM t-statistic

    Plots the mean value curve and standard deviation clouds of data that would
    be used as input to a function like `spm.get_spm_t_statistic()` to compute
    an SPM t-statistic.

    SPM t-statistic curve `spm_t_statistic` and a summary of
    the accompanying inference results `spm_t_inference` on the inputted
    Matplotlib axes object `ax`. This function modifies `ax` directly and does
    not return a new axis.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        Matplotlib axis object on which to plot
    group1 : ndarray
        2D Numpy array holding at least two time series, as documented in
        `spm.get_spm_t_statistic()`
    group2 : ndarray
        2D Numpy array holding at least two time series, as documented in
        `spm.get_spm_t_statistic()`
    t : ndarray, optional
        1D Numpy array holding the time (or other independent variable) values
        on which the time series in `group1` and `group2` are defined.

    Keyword Arguments
    -----------------
    xlabel : str
        Label for x axis
    ylabel : str
        Label for y axis
    title : str
        Axis title
    color1 : str
        Color of the mean value line of the data in `group1`.
    color2 : str
        Color of the mean value line of the data in `group2`.
    linewidth1 : str
        Width of the mean value line of the data in `group1`.
    linewidth2 : str
        Width of the mean value line of the data in `group2`.
    fillcolor1 : str
        Fill color of the standard deviation cloud of the data in `group1`.
    fillcolor2 : str
        Fill color of the standard deviation cloud of the data in `group2`.
    alpha1 : str
        Alpha of the standard deviation cloud of the data in `group1`.
    alpha2 : str
        Alpha of the standard deviation cloud of the data in `group2`.
    label1 : str
        Human-readable label for the `group1` data.
    label2 : str
        Human-readable label for the `group2` data.

    """
    if t is None:
        t = np.arange(y)

    xlabel=kwargs.get('xlabel', PlottingConstants.SPM_STATISTIC_DEFAULTS['xlabel'])
    ylabel=kwargs.get('ylabel', PlottingConstants.SPM_STATISTIC_DEFAULTS['ylabel'])
    title=kwargs.get('title', PlottingConstants.SPM_STATISTIC_DEFAULTS['title'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _remove_spines(ax)

    mean1 = np.mean(group1, axis=1)
    mean2 = np.mean(group2, axis=1)
    std1 = np.std(group1, ddof=1, axis=1)
    std2 = np.std(group2, ddof=1, axis=1)


    color1 = kwargs.get('color1', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['color1'])
    color2 = kwargs.get('color2', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['color2'])
    fillcolor1 = kwargs.get('fillcolor1', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['color1'])
    fillcolor2 = kwargs.get('fillcolor2', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['color2'])
    alpha1 = kwargs.get('alpha1', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['alpha1'])
    alpha2 = kwargs.get('alpha2', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['alpha2'])
    linewidth1 = kwargs.get('linewidth1', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['linewidth'])
    linewidth2 = kwargs.get('linewidth2', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['linewidth'])
    label1 = kwargs.get('label1', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['label1'])
    label2 = kwargs.get('label2', PlottingConstants.SPM_INPUT_DATA_DEFAULTS['label2'])

    # Mean value lines
    ax.plot(t, mean1, color=color1, linewidth=linewidth1, label=label1, zorder=PlottingConstants.SPM_INPUT_DATA_DEFAULTS['z_line1'])
    ax.plot(t, mean2, color=color2, linewidth=linewidth2, label=label2, zorder=PlottingConstants.SPM_INPUT_DATA_DEFAULTS['z_line2'])


    # Standard deviation clouds
    ax.fill_between(t, mean1 - std1, mean1 + std1, color=fillcolor1,
                    alpha=alpha1,
                    zorder=PlottingConstants.SPM_INPUT_DATA_DEFAULTS['z_fill1'])
    ax.fill_between(t, mean2 - std2, mean2 + std2, color=fillcolor2,
                    alpha=alpha2,
                    zorder=PlottingConstants.SPM_INPUT_DATA_DEFAULTS['z_fill2'])

    # Dashed line at y = 0
    ax.axhline(y=0, color=PlottingConstants.SPM_INPUT_DATA_DEFAULTS['x_axis_color'], linestyle=PlottingConstants.SPM_INPUT_DATA_DEFAULTS['x_axis_linestyle'])

    if label1 is not None or label2 is not None:
        ax.legend(framealpha=PlottingConstants.SPM_INPUT_DATA_DEFAULTS['legend_alpha'])


def _choose_cluster_to_display(clusters):
    """Choose an SPM suprathreshold cluster to highlight in plot.

    Chooses an SPM suprathreshold cluster whose parameters to display in an SPM
    plot—for lack of space, only one cluster can be displayed in the plot.

    The idea is to choose the "most significant" cluster; in the current
    implementation this is the cluster with the smallest p value, with largest
    cluster area used as a tie-breaker in the event that multiple clusters have
    equal p-values (in practice if multiple clusters have `p == 0.0`).

    Parameters
    ----------
    clusters : list
        List of SpmCluster namedtuples.

    Returns
    -------
    idx : int
        Indext of the cluster in `clusters` to display.

    """
    idx = 0
    for (i, cluster) in enumerate(clusters):
        if cluster.p < clusters[idx].p:
            idx = i
        elif math.isclose(cluster.p, clusters[idx].p):
            if cluster.area > clusters[idx].area:
                idx = i
    return idx


def _get_spm_axis_text(alpha, threshold, cluster):
    """
    Used to create a nice-looking string description of the parameters in an
    SPM suprathreshold cluster and a few global inference parameters. This
    string is then passed to Matplotlib's `ax.text`.
    
    The output looks something like...
    t* = 4.42
    alpha = 0.05
    p < 0.0001

    Parameters
    ----------
    alpha : float
        alpha value used for SPM inference.
    threshold : float
        t-statistic significance threshold produced by SPM inference.
    cluster : SpmCluster
        An SpmCluster named tuple whose parameters to display; this should be
        the "most significant" cluster produced by SPM inference.

    """
    # If no parameters were passed or if no supra-threshold clusters occurred,
    # write only t-star and alpha.
    if cluster is None:
        return "$\\alpha = {:.2f}$\n$t^* = {:.2f}$".format(threshold, alpha)

    # Parameters for cluster with smallest p-value
    p             = cluster.p
    start_time    = cluster.start_time
    end_time      = cluster.end_time
    extremum_time = cluster.extremum_time
    extremum      = cluster.extremum
    area          = cluster.area

    # The if/else block ensures p-value is written in nicely-readable
    # scientific notation.
    # See https://stackoverflow.com/a/45359185
    if p == 0.0:
        return str.format((
             "$t^* = {threshold:.2f}$\n"
            + "$\\alpha = {alpha:.2f}$\n"
            + "$p < 10^{{-16}}$\n"
            + "$T_1 = {start_time:.1f} \, \\mathrm{{ms}}$\n"
            + "$T_2 = {end_time:.1f} \, \\mathrm{{ms}}$\n"
            + "$t$-$\\mathrm{{max}} = {extremum:.1f}$\n"
            + "$T_{{\\mathrm{{max}}}} = {extremum_time:.1f} \, \\mathrm{{ms}}$\n"
            + "$\\mathrm{{Area}} = {area:.0f}$"
            ),
        alpha = alpha,
        threshold = threshold,
        start_time = start_time,
        end_time = end_time,
        extremum = extremum,
        extremum_time = extremum_time,
        area = area)
    else: # extract exponent and mantissa for scientific notation
        (sign, digits, exponent) = Decimal(p).as_tuple()
        exp = len(digits) + exponent - 1
        man = Decimal(p).scaleb(-exp).normalize()

        return str.format((
              "$t^* = {threshold:.2f}$\n"
            + "$\\alpha = {alpha:.2f}$\n"
            + "$p = {man:.0f} \\cdot 10^{{{exp:.0f}}}$\n"
            + "$T_1 = {start_time:.1f} \, \\mathrm{{ms}}$\n"
            + "$T_2 = {end_time:.1f} \, \\mathrm{{ms}}$\n"
            + "$t$-$\\mathrm{{max}} = {extremum:.1f}$\n"
            + "$T_{{\\mathrm{{max}}}} = {extremum_time:.1f} \, \\mathrm{{ms}}$\n"
            + "$\\mathrm{{Area}} = {area:.0f}$"
            ),
                          alpha = alpha,
                          threshold = threshold,
                          man = man,
                          exp = exp,
                          start_time = start_time,
                          end_time = end_time,
                          extremum = extremum,
                          extremum_time = extremum_time,
                          area = area)
        

def _remove_spines(ax):
    """Removes spines from Matplotlib axis for aesthetic purposes.

    Simple auxiliary function to remove upper and right spines from the
    inputted Matplotlib axis object; this looks cleaner, in my opinion.

    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
