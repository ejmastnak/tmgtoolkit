"""
Functions for reading data from and writing data to measurement files, and for
preprocessing of data read from measurement files in preparation for passing
the data to analysis functions.
"""

import numpy as np
import pandas as pd

from .constants import IoConstants

def tmg_excel_to_ndarray(fname, skiprows=None, nrows=None, skipcols=None, ncols=None):
    """Extracts information in a TMG measurement Excel file.

    Returns a TmgExcel namedtuple holding the information in a standard-format
    TMG measurement Excel file, as produced by the official TMG measurement
    software distributed with the TMG S1 and S2 measurement systems.

    Parameters
    ----------
    fname : string
        Path to a TMG measurement Excel file.

    Returns
    -------
    excel : TmgExcel
        A TmgExcel namedtuple holding the data in the Excel file. The TmgExcel
        namedtuple has the following fields:
        - `data` (ndarray): 2D Numpy array holding the TMG signals in the
              inputted Excel file. Measurements are stored in columns, so that
              `data` has shape `(rows, cols)`, where `rows` is the number of
              data points in each TMG measurement and `cols` is the number of
              measurements in the Excel file. Typically `rows` will be 1000,
              since a standard TMG signal is sampled for 1000 ms at 1 kHz.
    """
    if skiprows is None:
        skiprows = IoConstants.TMG_EXCEL_MAGIC_VALUES['data_start_row_idx']
    if nrows is None:
        nrows = IoConstants.TMG_EXCEL_MAGIC_VALUES['data_nrows']
    if skipcols is None:
        skipcols = IoConstants.TMG_EXCEL_MAGIC_VALUES['data_start_col_idx']

    usecols = lambda col: col >= skipcols and ((col < (ncols + skipcols)) if ncols is not None else True)
    return pd.read_excel(fname, header=None, skiprows=skiprows, nrows=nrows, usecols=usecols).values


def split_data_for_spm(data, numsets, n1, n2, split_mode, skiprows=0, nrows=None, equalize_columns=True):
    """Splits structured input data into groups for analysis with SPM.

    Splits the time series in the inputted 2D array `data` into groups that can
    then be compared to each other with SPM analysis. The function assumes
    `data` has a well-defined structure, namely that the time series in `data`
    are divided into `numsets` sets, where each set consists of `n1`
    consecutive time series in Group 1 followed by `n2` consecutive time series
    in Group 2.

    For conventional split modes splits `data` into an array of tuples
    `(group1, group2)` tuples, with one tuple for each measurement set in
    `data`.

    For "all"-type split modes, splits `data` into two 2D arrays stored in a
    single tuple `(group1, group2)`.

    For documentation of split modes see
    `constants.IoConstants.SPM_SPLIT_MODES`.

    Parameters
    ----------
    data : ndarray
        2D Numpy array holding time series data. The time series should be
        stored in columns, so that `data` has shape `(rows, cols)`, where
        `rows` is the number of data points in each time series measurement and
        `cols` is the number of time series.
    numsets : int
        Number of sets in `data`.
    n1 : int
        Number of Group 1 time series in each set.
    n2 : int
        Number of Group 2 time series in each set.
    skiprows : int, optional
        Skips the first `skiprows` in `data`.
    split_mode : int
        An symbolic constant from `constants.IoConstants` controlling how to
        split the measurements in `data`.
    nrows : int, optional
        If provided, return only the first `nrows` after `skiprows` in `data`.
        The default is to return all rows in `data`.
    equalize_columns : boolean, optional
        If True, all returned `group1` and `group2` arrays are guaranteed to
        have the same shape. If the inputted data does not split into an equal
        number of Group 1 and Group 2 time series under the given parameters,
        then the group with fewer time series is padded with additional time
        series until `group1` and `group2` have the same shape.

    Returns
    -------
    data_tuples : list
        Array holding a `(group1, group2)` tuple for each measurement set
        analyzed in `data`, where `group1` and `group2` are 2D Numpy arrays
        holding the Group 1 and Group 2 measurements, respectively, for each
        set. This return type is used for conventional split modes.
    data_tuple : tuple
        Tuple `(group1, group2)` holding Group 1 and Group 2 measurements.
        Fields are
        0 (group1) : ndarray
            2D Numpy array holding Group 1 measurements.
        1 (group2) : ndarray
            2D Numpy array holding Group 2 measurements.
        This return type is used for "all"-type split modes.

    """
    assert len(data.shape) == 2, "Inputted data must be two-dimensional array."
    assert skiprows >= 0, "The number of rows to skip must be non-negative."
    assert skiprows < data.shape[0], "The number of rows to skip ({}) exceeds the number of data rows ({}).".format(skiprows, data.shape[0])
    if nrows is not None:
        assert nrows > 0, "The requested number of rows to return must be greater than zero."
        assert nrows < data.shape[0] - skiprows, "The requested number of rows to return exceeds the number of data rows ({})".format(nrows, data.shape[0]) + ("and rows to skip ({}).".format(skiprows) if skiprows > 0 else ".")
    else:
        nrows = data.shape[0] - skiprows

    if split_mode == IoConstants.SPM_SPLIT_MODES['parallel']:
        return _split_data_parallel(data, numsets, n1, n2, skiprows, nrows, equalize_columns)
    elif split_mode == IoConstants.SPM_SPLIT_MODES['parallel_all']:
        return _split_data_parallel_all(data, numsets, n1, n2, skiprows, nrows, equalize_columns)
    elif split_mode == IoConstants.SPM_SPLIT_MODES['fixed_baseline']:
        return _split_data_fixed_baseline(data, numsets, n1, n2, skiprows, nrows, equalize_columns)
    elif split_mode == IoConstants.SPM_SPLIT_MODES['fixed_baseline_all']:
        return _split_data_fixed_baseline_all(data, numsets, n1, n2, skiprows, nrows, equalize_columns)
    elif split_mode == IoConstants.SPM_SPLIT_MODES['potentiation_creep']:
        return _split_data_potentiation_creep(data, numsets, n1, n2, skiprows, nrows, equalize_columns)
    elif split_mode == IoConstants.SPM_SPLIT_MODES['potentiation_creep_all']:
        return _split_data_potentiation_creep_all(data, numsets, n1, n2, skiprows, nrows, equalize_columns)
    else:
        raise ValueError("Unsupported split_mode ({}) passed to `split_data_for_spm`.".format(split_mode))


def _split_data_parallel(data, numsets, n1, n2, skiprows, nrows, equalize_columns):
    """Handles splitting SPM data for split mode `parallel`.

    For documentation of split modes see
    `constants.IoConstants.SPM_SPLIT_MODES`.

    Parameters
    ----------
    See `split_data_for_spm`.

    Returns
    -------
    data_tuples : list
        Length `numsets` array of tuples holding Group 1 and Group 2
        measurements for each measurement set.

    """
    data_tuples = []
    n = n1 + n2

    for s in range(numsets):
        idxs1 = list(range(s*n, s*n + n1))
        idxs2 = list(range(s*n + n1, (s + 1)*n))
        group1 = data[skiprows:skiprows + nrows, idxs1]
        group2 = data[skiprows:skiprows + nrows, idxs2]
        if equalize_columns:
            group1, group2 = _equalize_columns(group1, group2)
        data_tuples.append((group1, group2))

    return data_tuples


def _split_data_parallel_all(data, numsets, n1, n2, skiprows, nrows, equalize_columns):
    """Handles splitting SPM data for split mode `parallel_all`.

    For documentation of split modes see
    `constants.IoConstants.SPM_SPLIT_MODES`.

    Parameters
    ----------
    See `split_data_for_spm`.

    Returns
    -------
    data_tuple : tuple
        Tuple holding Group 1 and Group 2 measurements.

    """
    idxs1 = []
    idxs2 = []
    n = n1 + n2
    for s in range(numsets):
        idxs1.extend(range(s*n, s*n + n1))
        idxs2.extend(range(s*n + n1, (s + 1)*n))

    group1 = data[skiprows:skiprows + nrows, idxs1]
    group2 = data[skiprows:skiprows + nrows, idxs2]
    if equalize_columns:
        group1, group2 = _equalize_columns(group1, group2)
    return (group1, group2)


def _split_data_fixed_baseline(data, numsets, n1, n2, skiprows, nrows, equalize_columns):
    """Handles splitting SPM data for split mode `fixed_baseline`.

    For documentation of split modes see
    `constants.IoConstants.SPM_SPLIT_MODES`.

    Parameters
    ----------
    See `split_data_for_spm`.

    Returns
    -------
    data_tuples : list
        Length `numsets` array of tuples holding Group 1 and Group 2
        measurements for each measurement set.

    """
    data_tuples = []
    n = n1 + n2
    idxs1 = list(range(n1))  # measurements in first set only
    group1 = data[skiprows:skiprows + nrows, idxs1]
    
    for s in range(numsets):
        idxs2 = list(range(s*n + n1, (s + 1)*n))
        group2 = data[skiprows:skiprows + nrows, idxs2]
        if equalize_columns:
            group1, group2 = _equalize_columns(group1, group2)
        data_tuples.append((group1, group2))

    return data_tuples


def _split_data_fixed_baseline_all(data, numsets, n1, n2, skiprows, nrows, equalize_columns):
    """Handles splitting SPM data for split mode `fixed_baseline_all`.

    For documentation of split modes see
    `constants.IoConstants.SPM_SPLIT_MODES`.

    Parameters
    ----------
    See `split_data_for_spm`.

    Returns
    -------
    data_tuple : tuple
        Tuple holding Group 1 and Group 2 measurements.

    """
    idxs1 = []
    idxs2 = []
    n = n1 + n2

    idxs1.extend(range(n1))  # measurements in first set only
    for s in range(numsets):
        idxs2.extend(range(s*n + n1, (s + 1)*n))

    group1 = data[skiprows:skiprows + nrows, idxs1]
    group2 = data[skiprows:skiprows + nrows, idxs2]

    # No need to deal with adding noise if not equalizing columns
    if not equalize_columns:
        return (group1, group2)

    group1, group2 = _equalize_columns(group1, group2)

    # Apply noise to each Group 1 measurement beyond set 1. The assumption here
    # is that in fixed_baseline mode columns would have been added to Group 1
    # to match the number of columns in Group 2, but the `if` allows for the
    # edge case where Group 1 originally had more columns than Group 2.
    noise_cols = group2.shape[1] - n1
    if noise_cols <= 0:
        return (group1, group2) 

    mu = 0
    sigma = np.std(group2, ddof=1, axis=1)
    noise = np.zeros((group1.shape[0], noise_cols))
    for i in range(noise.shape[0]):
        noise[i] = np.random.default_rng().normal(mu, sigma[i], noise_cols)

    # Scale down noise to a small fraction of group1 peak-to-peak amplitude
    noise *= IoConstants.NOISE_SCALE * (np.max(group1) - np.min(group1))
    group1[:, n1:] += noise
    return (group1, group2) 


def _split_data_potentiation_creep(data, numsets, n1, n2, skiprows, nrows, equalize_columns):
    """Handles splitting SPM data for split mode `potentiation_creep`.

    For documentation of split modes see
    `constants.IoConstants.SPM_SPLIT_MODES`.

    Parameters
    ----------
    See `split_data_for_spm`.

    Returns
    -------
    data_tuples : list
        Length `numsets - 1` array of tuples holding Group 1 and Group 2
        measurements for each measurement set.

    """
    data_tuples = []
    n = n1 + n2
    idxs1 = list(range(n1))  # measurements in first set only
    group1 = data[skiprows:skiprows + nrows, idxs1]

    for s in range(1, numsets):
        idxs2 = list(range(s*n, s*n + n1))  # later sets of Group 1
        group2 = data[skiprows:skiprows + nrows, idxs2]
        if equalize_columns:
            group1, group2 = _equalize_columns(group1, group2)
        data_tuples.append((group1, group2))
    
    return data_tuples


def _split_data_potentiation_creep_all(data, numsets, n1, n2, skiprows, nrows, equalize_columns):
    """Handles splitting SPM data for split mode `potentiation_creep_all`.

    For documentation of split modes see
    `constants.IoConstants.SPM_SPLIT_MODES`.

    Parameters
    ----------
    See `split_data_for_spm`.

    Returns
    -------
    data_tuple : tuple
        Tuple holding Group 1 and Group 2 measurements.

    """
    idxs1 = []
    idxs2 = []
    n = n1 + n2

    idxs1.extend(range(n1))  # measurements in first set only
    for s in range(1, numsets):  # later sets of Group 1
        idxs2.extend(range(s*n, s*n + n1))

    group1 = data[skiprows:skiprows + nrows, idxs1]
    group2 = data[skiprows:skiprows + nrows, idxs2]

    # No need to deal with adding noise if not equalizing columns
    if not equalize_columns:
        return (group1, group2)

    group1, group2 = _equalize_columns(group1, group2)

    # Apply noise to each Group 1 measurement beyond set 1. The assumption here
    # is that the original Group 1 will have had multiple measurement sets
    # beyond set 1, but the `if` allows for the edge case where the original
    # Group 1 originally had 1 or 2 measurement sets, in which case `group1`
    # and `group2` would have the same number of columns.
    noise_cols = group2.shape[1] - n1
    if noise_cols <= 0:
        return (group1, group2) 

    mu = 0
    sigma = np.std(group2, ddof=1, axis=1)
    noise = np.zeros((group1.shape[0], noise_cols))
    for i in range(noise.shape[0]):
        noise[i] = np.random.default_rng().normal(mu, sigma[i], noise_cols)

    # Scale down noise to a small fraction of group1 peak-to-peak amplitude
    noise *= IoConstants.NOISE_SCALE * (np.max(group1) - np.min(group1))
    group1[:, n1:] += noise
    return (group1, group2) 


def _equalize_columns(group1, group2):
    """Ensures inputted 2D arrays have the same number of columns.

    Context: 2D arrays inputted to `spm1d`'s analysis functions require that
    the arrays have the same number of rows and columns. This function
    equalizes the number of columns in the inputted data, if necessary, so that
    the data can be analyzed with `spm1d`.

    The function works by computing the mean of the array with fewer columns,
    and repeatedly appending this mean column to the array with fewer columns
    until the number of columns in `group1` and `group2` are equal.

    Parameters
    ----------
    group1 : ndarray
        2D Numpy array holding at least two time series.
    group2 : ndarray
        2D Numpy array holding at least two time series.

    Returns
    -------
    group_tuple : tuple
        Tuple holding equalized versions of `group1` and `group2`; fields are
        0. `group1` (ndarray)
        1. `group2` (ndarray)
    """
    rows = group1.shape[0]
    cols1 = group1.shape[1]
    cols2 = group2.shape[1]
    if cols1 == cols2:
        return (group1, group2)

    elif cols1 < cols2:
        padded_group1 = np.zeros((rows, cols2))
        padded_group1[:, 0:cols1] = group1
        mean1 = np.mean(group1, axis=1)
        for c in range(cols1, cols2):
            padded_group1[:, c] = mean1
        return (padded_group1, group2)
    elif cols2 < cols1:
        padded_group2 = np.zeros((rows, cols1))
        padded_group2[:, 0:cols2] = group2
        mean2 = np.mean(group2, axis=1)
        for c in range(cols2, cols1):
            padded_group2[:, c] = mean2
        return (group1, padded_group2)
