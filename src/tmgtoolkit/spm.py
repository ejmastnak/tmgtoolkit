from collections import namedtuple
import numpy as np
import spm1d

from .constants import SpmConstants, NamedTupleTypes
from .time_series import _idx_to_time, _interpolate_extremum

def get_spm_t_statistic(group1, group2, mitigate_iir_filter_artefact=True, swap_groups=False):
    """Computes SPM t-test statistic for the inputted data.

    Returns an SpmTStatistic namedtuple containing the 1D SPM t-test statistic
    resulting from a paired SPM t-test comparing the time series data in the
    inputted arrays `group1` and `group2`. The returned SPM t-statistic is
    implicitly defined on the same time (or other independent variable) grid as
    the time series in `group1` and `group2`, but it is up to the calling code
    to keep track of these time values.

    Parameters
    ----------
    group1 : ndarray
        2D Numpy array holding at least two time series. Each of the time
        series in `group1` should be the same length, and each series should be
        stored as a column in `group1`, so that `group1` has shape
        (points_per_series, number_of_series).
    group2 : ndarray
        2D Numpy array holding at least two time series, to be compared to the
        set of time series in `group1`. The number of time series in `group2`
        may be different from the number of time series in `group1`, but the
        length of the time series in `group1` and `group2` should be equal
        (i.e. `group1` and `group2` should have the same number of rows).
    mitigate_iir_filter_artefact : bool, optional
        If True, passes data through `_mitigate_iir_filter_artefact` before
        computing the SPM t-statistic. See `_mitigate_iir_filter_artefact` for
        details.
    swap_groups : bool, optional
        Controls the order in which `group1` and `group2` are passed to spm1d's
        paired t-test function. This can be useful e.g. if client wants to
        manipulate which group is treated as "potentiated".

    Pre-Conditions
    --------------
    1. `group1` and `group2` must have the same shape—this is a requirement for
       further analysis by spm1d.
    2. No row in `group1` or in `group2` can have all equal values—this is to
       ensure there are no rows in the inputted data with zero variance, which
       would cause a divide by zero error when computing the SPM t-statistic.
       This is again a requirement for further analysis by spm1d.

    Note on time values: although the time (or other independent variable)
    values on which `group1` and `group2` are defined are not needed to compute
    the SPM t-statistic, `group1` and `group2` should conceptually be defined
    on the same time grid; this becomes relevant when performing inference on
    the t-statistic returned by this function.

    Returns
    -------
    spm_ts : SpmTStatistic
        A SpmTStatistic namedtuple holding the SPM t-test statistic curve
        produced by comparing the time series data in the inputted arrays `group1`
        and `group2`.
        The SpmTStatistic namedtuple has a single field:
        - `t_statistic` (ndarray): a 1D Numpy array holding the SPM t-statistic
              curve resulting from the comparison of the time series in `group1` to
              the time series in `group2`. The length of `t_statistic` will equal
              the length of the time series in `group1` and `group2` (or the length of
              the shorter of the time series in `group1` and `group2` if the series in
              `group1` and `group2` have different lengths).
        - `spm_t` (spm1d._spm.SPM_T): wrapper object for the t-statistic used
            by the spm1d library. Meant for internal use only and subject to
            change in future versions.
        Access fields with e.g. `spm_ts.t_statistic`.

    """
    if group1.shape != group2.shape:
        raise ValueError("Group 1 and Group 2 must have the same shape, but have shapes {} and {}.".format(group1.shape, group2.shape))

    if mitigate_iir_filter_artefact:
        group1, group2 = _mitigate_iir_filter_artefact(group1, group2)
    if swap_groups:
        spm_t = spm1d.stats.ttest_paired(group2.T, group1.T)
    else:
        spm_t = spm1d.stats.ttest_paired(group1.T, group2.T)
    return NamedTupleTypes.SpmTStatistic(t_statistic=spm_t.z, spm_t=spm_t)


def get_spm_t_inference(spm_ts, t=None, alpha=0.05, two_tailed=True):
    """Performs SPM inference on the inputted SPM t-statistic data.

    Returns an SpmTInference namedtuple holding the results of performing
    inference on the inputted SpmTStatistic namedtuple at the Type I error
    level `alpha`.

    Parameters
    ----------
    spm_ts : SpmTStatistic
        A SpmTStatistic namedtuple as returned by `get_spm_t_statistic`.
    t : ndarray, optional
        1D Numpy array holding the time (or other independent variable) values
        on which the SPM t-statistic curve `spm_ts.t_statistic` is
        defined. `t` is used to return inference-related time parameters in
        correct units.
        If provided, `t` must be a 1N Numpy array with the same number of
        points as `spm_ts.t_statistic` (this is the same time grid on
        which the time series used to compute `spm_ts` were defined).
        If not provided, this function will use index values of
        `spm_ts.t_statistic` as the independent variable, i.e. `t =
        [0, 1, 2, ..., spm_ts.t_statistic.shape[0]]`.
    alpha : float
        Type I error rate (probability of rejecting the null hypothesis given
        that it is true) when performing inference.
    two_tailed : bool
        Whether to perform two-sided or one-sided inference.

    Returns
    -------
    spm_t_inference : SpmTInference
        A SpmTInference namedtuple produced by performing inference on the
        inputted SpmTStatistic namedtuple `spm_ts`.
        The SpmTStatistic namedtuple has the following fields:
        - `alpha` (float): alpha used for inference.
        - `p` (float): p value for entire inference.
        - `threshold` (float): SPM t-statistic significance threshold value.
        - `clusters` (list): a list of SpmCluster namedtuples summarizing each
              supra-threshold cluster, or an empty list if the inference did
              not produce supra-threshold clusters. Each SpmCluster namedtuple
              has the following fields:
              - `idx` (int): the cluster's 0-based index within `clusters`
              - `p` (float): the cluster's p value.
              - `start_time` (float): time at which the cluster begins, in the
                  same units as `t`.
              - `end_time` (float): time at which the cluster ends, in the same
                    units as `t`.
              - `centroid_time` (float): time of the cluster's centroid, in the
                    same units as `t`.
              - `centroid` (float): SPM t-statistic value of the cluster's
                centroid.
              - `extremum_time` (float): time of the cluster's extremum (which
                    can in general be either a maximum or a minimum), in the
                    same units as `t`.
              - `extremum` (float): SPM t-statistic value of the cluster's
                extremum.
              - `area` (float): the cluster's area, i.e. area of the region
                    bounded by the SPM t-statistic curve and the horizontal
                    line at the significance threshold
                    `spm_t_inference.threshold` from the cluster's `start_time`
                    to the cluster's `end_time`.
        Access fields with e.g. `spm_t_inference.p` for the value of `p`.
    """
    spm_ti = spm_ts.spm_t.inference(alpha=alpha, two_tailed=two_tailed, interp=True)

    if t is None:
        t = np.arange(spm_ts.t_statistic)

    return NamedTupleTypes.SpmTInference(alpha=alpha, p=spm_ti.p_set,
                                         threshold=spm_ti.zstar,
                                         clusters=_get_spm_clusters(spm_ti, t))


def _mitigate_iir_filter_artefact(group1, group2):
    """Mitigates artefact from IIR filtering of TMG signals.

    Context: standard TMG measurements produced by a TMG S1 or S2 measurement
    system are IIR filtered, which leaves a small wave-like artefact over the
    first few milliseconds of the signal. This artefact is negligible in most
    cases—the artefact amplitudes are orders of magnitude smaller than typical
    TMG signal values—but is picked up by SPM, which then reports
    non-physically large values of the SPM t-statistic over the first
    milliseconds of an t-statistic curve produced by comparing TMG signals. The
    goal of this function is to mitigate the IIR artefact enough that SPM will
    not report statisticaly significant t-statistic values over the first few
    milliseconds of a TMG signal.

    This function mitigates the IIR artefact very simply, by subtracting from
    the post-conditioning data the average of the difference between pre- and
    post-conditioning data over the first few ms of the TMG curve; see
    implementation details in the code below. This is safe to do without
    appreciable affecting the later (i.e. t > ~5 ms) portion of a TMG signal
    because signal values in the first few milliseconds are of the order 0.001
    mm, while typical values later in the TMG curve are of the order 5 mm.

    Parameters
    ----------
    group1 : ndarray
        2D Numpy array holding at least two time series, as for
        `get_spm_t_statistic`.
    group2 : ndarray
        2D Numpy array holding at least two time series, as for
        `get_spm_t_statistic`.

    Returns
    -------
    group_tuple : tuple
        Tuple holding mitigated versions of `group1` and `group2`; fields are
        0. `group1` (ndarray)
        1. `group2` (ndarray)
    """
    # Mean is taken over rows (time) and then over columns (time series) to
    # produce a single scalar value for each 2D ndarray.
    nrows = SpmConstants.T_STATISTIC['iir_artefact_mitigation_rows']
    mean1 = np.mean(np.mean(group1[:nrows, :], axis=0))
    mean2 = np.mean(np.mean(group2[:nrows, :], axis=0))
    if mean2 > mean1:
        group2 = group2 - np.mean(mean2 - mean1)
    else:
        group1 = group1 - np.mean(mean1 - mean2)
    return (group1, group2)


def _get_spm_clusters(spm_ti, t):
    """Returns information describing an SPM inference object's clusters.

    Returns a list of SpmCluster namedtuples summarizing the supra-threshold
    clusters in the SPM inference object `spm_ti`.

    Parameters
    ----------
    spm_ti : spm1d._spm.SPMi_T
        An SPMi_T inference object, as returned by the spm1d library when
        performing inference on a SPM_T object.
    t : ndarray
        1D Numpy array holding the time (or other independent variable) values
        on which the SPM t-statistic curve used to compute the inference object
        `spm_ti` is defined. See `get_spm_t_inference` for details.

    Returns
    -------
    clusters : list
        A list of SpmCluster namedtuples summarizing the supra-threshold
        clusters in `spm_ti`, or an empty list if the inference did not produce
        and clusters. See `get_spm_t_inference` for documentation of SpmCluster
        fields.
        
    """
    # See _get_params_of_spm_cluster in frontiers/src/spm_analysis.py
    if spm_ti.clusters is None:
        return []

    clusters = []
    for idx, cluster in enumerate(spm_ti.clusters):
        clusters.append(_analyze_spm1d_cluster(cluster, t, idx))
    return clusters


def _analyze_spm1d_cluster(cluster, t, idx):
    """Returns an SpmCluster namedtuple summarizing an SPM Cluster object.

    Parameters
    ----------
    spm1d_cluster : spm1d._clusters.Cluster
        An spm1d Cluster object representing a suprathreshold cluster
        associated with an SPM inference.
    t : ndarray
        1D Numpy array holding the time (or other independent variable) values
        on which the SPM t-statistic curve used to compute the inference object
        `spm_ti` is defined. See `get_spm_t_inference` for details.
    idx : int
        The cluster's 0-based index within its parent `clusters` list.

    Returns
    ----------
    cluster : SpmCluster
        An SpmCluster namedtuple summarizing the inputted
        spm1d._clusters.Cluster object. See `get_spm_t_inference` for
        documentation of SpmCluster fields.
        
    """
    start_idx, end_idx = cluster.endpoints
    centroid_idx, centroid = cluster.centroid
    threshold = cluster.threshold

    idxs = cluster._X
    spmt = cluster._Z

    extremum_idx = np.argmax(np.abs(spmt))
    idx_window = []
    spmt_window = []
    padding = SpmConstants.CLUSTER['interpolation_window_padding']
    for i in range(extremum_idx - padding, extremum_idx + padding + 1):
        if i >= 0 and i < len(spmt):
            idx_window.append(i)
            spmt_window.append(spmt[i])
    extremum_idx, extremum = _interpolate_extremum(idx_window, spmt_window, True)

    area_above_x = np.trapz(np.abs(spmt), x=idxs)
    cluster_area = area_above_x - threshold*(idxs[-1] - idxs[0])

    start_time = _idx_to_time(start_idx, t)
    end_time = _idx_to_time(end_idx, t)
    centroid_time = _idx_to_time(centroid_idx, t)
    extremum_time = _idx_to_time(extremum_idx, t)

    return NamedTupleTypes.SpmCluster(idx=idx, p=cluster.P, start_time=start_time,
                                      end_time=end_time,
                                      centroid_time=centroid_time,
                                      centroid=centroid,
                                      extremum_time=extremum_time,
                                      extremum=extremum, area=cluster_area)
