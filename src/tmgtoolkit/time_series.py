import sys
import math
from collections import namedtuple
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ttest_rel
from scipy.interpolate import lagrange

from .constants import TimeSeriesConstants, NamedTupleTypes

def get_tmg_parameters_of_time_series(y, t=None,
                                      ignore_maxima_with_idx_less_than=None,
                                      ignore_maxima_less_than=None,
                                      use_first_max_as_dm=False,
                                      interpolate_dm=False):
    """Returns TMG parameters for a time series.

    Returns a TmgParams namedtuple holding the TMG parameters Dm, Td, Tc, Ts,
    and Tr for the inputted time series `y`.

    Parameters
    ----------
    y : ndarray
        1D Numpy array holding the values of a time series signal. Typically
        this will be a TMG signal of muscle displacement measured with respect
        to time.
    t : ndarray, optional
        1D Numpy array holding the time (or other independent variable) values
        on which `y` is defined. This is used to return the values of the time
        parameters Td, Tc, Ts, and Tr in correct units.

        If provided, `t` must be a 1N Numpy array with the same number of
        points as `y`. If not provided, this function will use index values of
        `y` as the independent variable, i.e. `t = [0, 1, 2, ..., y.shape[0]]`.

        Suggestion: If you are computing the TMG parameters of a standard TMG
        signal sampled at 1 kHz, you can leave `t` as `None` and rely on the
        default values `t = [0, 1, 2, ..., y.shape[0]]` (or, even better, be
        explicit and use e.g. `t=np.arange[y.shape[0]]`) and interpret the
        values of `t` as milliseconds; this works because in a standard TMG
        signal sampled at 1 kHz, the samples are uniformly spaced in time with
        spacing of 1 millisecond. The returned values of the parameters Td, Tc,
        Ts, and Tr will then be in milliseconds.
    ignore_maxima_with_idx_less_than : int, optional
        Ignore data points with index less than
        `ignore_maxima_with_idx_less_than` when computing Dm. Used in practice
        to avoid tiny maxima resulting from filtering artefacts in the first
        few milliseconds of a TMG signal. Will use a sane default value
        designed for TMG signals if no value is specified.
    ignore_maxima_less_than : float, optional
        Ignore data points with values less than `ignore_maxima_less_than` when
        computing Dm. Used in practice to avoid tiny maxima resulting from
        filtering artefacts in the first few milliseconds of a TMG signal. Will
        use a sane default value designed for TMG signals if no value is
        specified.
    use_first_max_as_dm : bool
        If True, uses the first maximum meeting the criteria imposed by
        `ignore_maxima_with_idx_less_than` and `ignore_maxima_less_than` for
        Dm; if false, uses the global maximum for Dm. Used in practice to make
        Dm, and TMG parameters derived from it, correspond to the twitch from
        fast-twitch muscle fibers, which may have a distinct, earlier maximum
        than the global maximum caused by slower-twitch fibers.
    interpolate_dm : bool
        If True, uses interpolation to fine-tune the value of Dm beyond the
        granularity of `y`'s discrete samples. If False, uses the maximum
        sample in `y` as Dm. See `_interpolate_extremum` for more context on
        interpolation.

    Returns
    -------
    params : TmgParams
        A TmgParams namedtuple holding the computed TMG parameter values. The
        TmgParams namedtuple has the following fields:
        - `dm` (float): value of Dm, in the same units as `y`.
        - `tm` (float): time of Dm, in the same units as `t`.
        - `td` (float): value of Td, in the same units as `t`.
        - `tc` (float): value of Tc, in the same units as `t`.
        - `ts` (float): value of Ts, in the same units as `t`.
        - `tr` (float): value of Tr, in the same units as `t`.
        Access fields with e.g. `params.dm` for value of `dm`.
    """
    if ignore_maxima_with_idx_less_than is None:
        ignore_maxima_with_idx_less_than = TimeSeriesConstants.TMG_PARAMS['ignore_maxima_with_idx_less_than']
    if ignore_maxima_less_than is None:
        ignore_maxima_less_than = TimeSeriesConstants.TMG_PARAMS['ignore_maxima_less_than']

    dm_idx, dm, float_dm_idx = _get_dm_idx_and_value(y,
                                                     ignore_maxima_with_idx_less_than,
                                                     ignore_maxima_less_than,
                                                     use_first_max_as_dm,
                                                     interpolate_dm)

    t10_left_idx = _interpolate_idx_of_target_amplitude(y, 0.1*dm, True)
    t50_left_idx = _interpolate_idx_of_target_amplitude(y, 0.5*dm, True)
    t90_left_idx = _interpolate_idx_of_target_amplitude(y, 0.9*dm, True)
    t90_right_idx = _interpolate_idx_of_target_amplitude(y, 0.9*dm, False, start_search_at_idx=dm_idx)
    t50_right_idx = _interpolate_idx_of_target_amplitude(y, 0.5*dm, False, start_search_at_idx=dm_idx)

    # Convert indices to time
    tm = _idx_to_time(float_dm_idx, t)
    t10_left = _idx_to_time(t10_left_idx, t)
    t50_left = _idx_to_time(t50_left_idx, t)
    t90_left = _idx_to_time(t90_left_idx, t)
    t90_right = _idx_to_time(t90_right_idx, t)
    t50_right = _idx_to_time(t50_right_idx, t)

    # Compute standard TMG time parameters
    td = t10_left
    tc = t90_left - t10_left
    ts = t50_right - t50_left
    tr = t50_right - t90_right

    return NamedTupleTypes.TmgParams(dm=dm, tm=tm, td=td, tc=tc, ts=ts, tr=tr)


def get_derivative_of_time_series(y, t=None):
    """Returns the derivative of a time series.

    Returns the derivative with respect to time of the inputted time series.

    Parameters
    ----------
    y : ndarray
        1D Numpy array holding the values of a time series signal, as for
        `get_tmg_parameters_of_time_series`.
    t : ndarray, optional
        1D Numpy array holding the time (or other independent variable) values
        on which `y` is defined.

        If provided, `t` must be a 1N Numpy array with the same number of
        points as `y`. If not provided, this function will use index values of
        `y` as the independent variable, i.e. `t = [0, 1, 2, ..., y.shape[0]]`.

    Returns
    -------
    dydt : ndarray
        1D Numpy array holding the derivative of the inputted time series `y`.
        The derivative `dydt` has the same dimensions as `y`, is defined on the
        same time grid `t` as `y`,  and the units of `dydt` are the units of
        `y` divided by the units of `t`.

    """
    if t is None:
        t = numpy.arange(y.shape[0])
    return np.gradient(y, t)


def get_extremum_parameters_of_time_series(y, t=None):
    """Returns extremum parameters of a time series.

    Returns a TmgExtremumParams namedtuple holding the maximum value, time of
    maximum value, minimum value, and time of minimum value of the inputted
    time series `y`.

    Parameters
    ----------
    y : ndarray
        1D Numpy array holding the values of a time series signal, as for
        `get_tmg_parameters_of_time_series`. Typically `y` will be either a TMG
        signal or a TMG signal's time derivative, (e.g. as computed by
        `get_derivative_of_time_series`).
    t : ndarray, optional
        1D Numpy array holding the time (or other independent variable) values
        on which `y` is defined.

        If provided, `t` must be a 1N Numpy array with the same number of
        points as `y`. If not provided, this function will use index values of
        `y` as the independent variable, i.e. `t = [0, 1, 2, ..., y.shape[0]]`.

    Returns
    -------
    params : TmgExtremumParams
        A TmgExtremumParams namedtuple holding the computed parameter values.
        The TmgExtremumParams namedtuple has the following fields:
        - `max` (float): maximum value of `y`, in the same units as `y`.
        - `max_time` (float): time at which `max` occurs, in the same units as
              `t`. If `y` has multiple equal maximum values, the time of the
              first maximum value is used.
        - `min` (float): minimum value of `y`, in the same units as `y`.
        - `min_time` (float): time at which `min` occurs, in the same units as
              `t`. If `y` has multiple equal minimum values, the time of the
              first minimum value is used.
        Access fields with e.g. `params.max` for the value of `max`.
        
        Note: the extremum values and their times are computed by interpolating
        `y` and `t`, and in general the maximum and minimum values will fall
        between between discrete values in `y`, and the maximum and minimum
        times will fall between discrete values in `t`—this is expected because
        of interpolation.
        When `t` is not provided, the maximum and minimum times (which will in
        generally be non-integer, floating point values) should be interpretted
        as the index values where `max` and `min` would fall if `y` where
        defined on a continuous index domain.

    """
    if t is None:
        t = np.arange(y.shape[0])

    # Construct window around maximum (as index bounds allow)
    max_idx_estimate = np.argmax(y)
    idx_window = []
    y_window = []
    padding = TimeSeriesConstants.EXTREMUM_PARAMS['interpolation_window_padding']
    for i in range(max_idx_estimate - padding, max_idx_estimate + padding + 1):
        if i >= 0 and i < len(y):
            idx_window.append(i)
            y_window.append(y[i])
    max_idx, max = _interpolate_extremum(idx_window, y_window, True)

    # Construct window around minimum (as index bounds allow)
    min_idx_estimate = np.argmin(y)
    idx_window = []
    y_window = []
    for i in range(min_idx_estimate - padding, min_idx_estimate + padding + 1):
        if i >= 0 and i < len(y):
            idx_window.append(i)
            y_window.append(y[i])
    min_idx, min = _interpolate_extremum(idx_window, y_window, False)

    max_time = _idx_to_time(max_idx, t)
    min_time = _idx_to_time(min_idx, t)

    return NamedTupleTypes.ExtremumParams(max_time=max_time, max=max, min_time=min_time, min=min)


def _get_dm_idx_and_value(y, ignore_maxima_with_idx_less_than,
                          ignore_maxima_less_than, use_first_max_as_dm,
                          interpolate_dm):
    """Returns index and value of Dm subject to TMG-specific constraints.

    Returns
    -------
    dm_tuple : tuple
        Tuple holding index and value of Dm. Keys are
        0 (int): index of Dm in `y`, cast to an int. Casting is relevant when
            using interpolation to find Dm, in which case the "index" would
            otherwise be a floating point value.
        1 (float): value of Dm.
        2 (float): index of Dm in `y`.  When using interpolation to find Dm,
            this is the estimated floating-point "index" where Dm would fall if
            `y` where continuous. When not interpolating, this field is
            equivalent to `dm_tuple[0]`.

    """
    max_idxs = find_peaks(y, height=ignore_maxima_less_than)[0]

    # Keep only maxima after ignore_maxima_with_idx_less_than
    max_idxs = max_idxs[(max_idxs > ignore_maxima_with_idx_less_than)]

    dm_idx = int(max_idxs[0])  # cast is from np.int64 to int
    if not use_first_max_as_dm:
        for candidate in max_idxs:
            if y[candidate] > y[dm_idx]:
                dm_idx = candidate
    dm = y[dm_idx]

    if interpolate_dm:
        idx_window = []
        y_window = []
        padding = TimeSeriesConstants.EXTREMUM_PARAMS['interpolation_window_padding']
        for i in range(dm_idx - padding, dm_idx + padding + 1):
            if i >= 0 and i < len(y):
                idx_window.append(i)
                y_window.append(y[i])
        dm_idx, dm = _interpolate_extremum(idx_window, y_window, True)

    return (int(dm_idx), dm, dm_idx)


def _interpolate_idx_of_target_amplitude(y, target, upcrossing, start_search_at_idx=0):
    """Estimate index at which a time series reaches target amplitude.

    Uses interpolation to estimate the "index" at which a time series first
    reaches a target amplitude with finer granularity than the time series'
    underlying sampling allows.

    Used in practice to estimate the time when a TMG signal *first* reaches an
    amplitude equal to a target percentage of Dm (e.g. 10%, 50%, 90% etc.);
    this is nontrivial because generally the target amplitude falls between
    samples of the TMG signal, and so must be interpolated.

    The function interpolates a Lagrange polynomial around the point of target
    amplitude, then uses a root-finding algorithm to find the "index" value
    where the interpolating polynomial crosses the target amplitude.

    Note: 
    The function returns the estimated "index" at which the inputted signal
    would *first* reach the target amplitude if the signal were continuous.

    Parameters
    ----------
    y : ndarray
        1D Numpy array, in practice holding a TMG signal.
    target : double
        Target amplitude to find the index of.
    upcrossing : bool
        Set to `True` to find index at which `y` first crosses above `target`;
        set to `False` to find index at which `y` first crosses below `target`.
    start_search_at_idx : int, optional
        Only begin searching for target amplitude from this index onward.

    Returns
    ----------
    idx : float
        Estimated index at which `y` first reaches `target` amplitude.

    """
    # Find the two points enclosing target amplitude
    left_idx = None
    right_idx = None
    for i in range(start_search_at_idx, len(y) - 1):
        if upcrossing:
            if y[i] <= target and y[i + 1] >= target:
                left_idx = i
                right_idx = i + 1
                break
        else:
            if y[i] >= target and y[i + 1] <= target:
                left_idx = i
                right_idx = i + 1
                break

    if left_idx is None or right_idx is None:
        print("Error: no maximum found!", file=sys.stderr)
        sys.exit(1)

    # Store index of point closest to target amplitude for use below after
    # root-finding algorithm.
    closest_idx_to_target = left_idx if abs(target - y[left_idx]) < abs(target - y[right_idx]) else right_idx

    # Create a window around target amplitude (as bounds permit)
    idx_window = []
    y_window = []
    padding = TimeSeriesConstants.TMG_PARAMS['interpolation_window_padding']
    for i in range(left_idx - padding, right_idx + padding + 1):
        if i >= 0 and i < len(y):
            idx_window.append(i)
            y_window.append(y[i])

    poly = lagrange(idx_window, y_window)
    coef = poly.coef

    # Subtract off target amplitude to prepare polynomial for use with a
    # root-finding algorithm
    coef[-1] -= target  

    # Points where interpolating polynomial equals target amplitude; cast to
    # real removes residual imaginary part left by the root-finding algorithm.
    roots = np.real(np.roots(coef))

    # The polynmial could in general have multiple roots; return the closest
    # root to the index of the target amplitude.
    return roots[np.argmin(np.abs(roots - closest_idx_to_target))]


def _idx_to_time(idx, t):
    """Convert floating-point "index" to time value.

    Converts a (in-general) non-integer "index" into a time array into the
    units of the time. Used in practice after interpolating a TMG signal
    produces points that fall between discrete samples in the signal.

    Parameters
    ----------
    idx : float
        In-general floating-point index into `t`.
    t : ndarray
        1D array of time (or other independent variable) values.

    Returns
    ----------
    time : float
        Approximate independent variable value, in units of `t`, corresponding
        to `idx`; loosely, an estimate of the value t(idx) if `t` were
        continuous.
    """
    assert 0 <= idx < t.shape[0], "Index {} is out of bounds for time array with shape {}.".format(idx, t.shape)

    if idx.is_integer():
        return float(t[int(idx)])

    idx_floor = math.floor(idx)
    idx_ceil = math.ceil(idx)
    ratio = (idx - idx_floor)/(idx_ceil - idx_floor)

    t_floor = t[idx_floor]
    t_ceil = t[idx_ceil]
    return t_floor + ratio * (t_ceil - t_floor)


def _interpolate_extremum(idx_window, y_window, find_max):
    """Estimates the index and value of a time series' extremum.

    Uses interpolation to estimate the index and value of a time series'
    extremum with finer granularity than the time series' underlying sampling
    allows.

    Used in practice to find the extremum and minimum values of a TMG signal's
    derivative; one just finds the extremum of a negated signal to find the
    original signal's minimum.

    The function interpolates a Lagrange polynomial around the discrete
    signal's extremum point, then estimates the more accurate, interpolated
    extremum by evaluating the interpolating polynomial on a finely-spaced
    index-domain grid, e.g. an order of magnitude finer than the spacing in
    `idx_window`, then finding the extremum value of the resulting points.

    Notes:
    - The interpolating polynomial's extremum is not found analytically, even
      though this is in principle possible. The extra precision is irrelevant
      and probably meaningless noise since the interpolation is an estimate
      anyway.
    - The entire TMG/RDD signal is passed in. This makes code a bit cleaner
      on the calling end, even if in principle one needs to pass in only a
      small window of index and displacement points centered around the extrema.

    Parameters
    ----------
    idx_window : ndarray
        1D Numpy array a small window of indices of time series data points
        around the series' extremum value.
    y_window : ndarray
        1D Numpy array a small window of time series data points around the
        extremum value.
    find_max : bool
        True to find maximum; False to find minimum.

    Returns
    -------
    extremum_tuple : tuple
        Tuple holding index and value of extremum. Fields are
        0 (float): estimated floating-point "index", in units of `idx_window`,
            at which the extremum in `y_window` would fall if `y_window` were
            continuous.
        1 (float): interpolated extremum in `y_window`.
    
    """
    poly = lagrange(idx_window, y_window)

    idx_grid = np.linspace(idx_window[0], idx_window[-1], len(idx_window)*TimeSeriesConstants.EXTREMUM_PARAMS['poly_grid_magnification'])
    y = poly(idx_grid)

    extremum_idx = np.argmax(y) if find_max else np.argmin(y)
    return idx_grid[extremum_idx], y[extremum_idx]
