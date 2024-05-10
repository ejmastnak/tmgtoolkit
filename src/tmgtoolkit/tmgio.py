"""
Functions for reading data from and writing data to measurement files, and for
preprocessing of data read from measurement files in preparation for passing
the data to analysis functions.
"""

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


def split_data_for_spm(data, numsets, n1, n2, nrows=None, split_mode=None):
    """Splits structured inputted data into two groups for analysis with SPM.

    Splits the time series in the inputted 2D array `data` into two groups, 1
    and 2, that can then be compared to each other with SPM analysis.

    The function assumes `data` has a well-defined structure, namely that the
    time series in `data` are divided into `numsets` sets, where each set
    consists of `n1` consecutive time series in group 1 followed by `n2`
    consecutive time series in group 2.

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
        Number of group 1 time series in each set.
    n2 : int
        Number of group 2 time series in each set.
    nrows : int, optional
        If provided, return only the first `nrows` in `data`. The default is to
        return all rows in `data`.
    split_mode : int, optional
        An symbolic constant from `constants.IoConstants` controlling how to
        split the measurements in `data`.

    Returns
    -------
    data_tuple : tuple
        Tuple holding group 1 and group 2 series. Fields are
        0 (group1) : ndarray
            2D Numpy array holding group 1 measurements.
        1 (group2) : ndarray
            2D Numpy array holding group 2 measurements.

    """
    if nrows is None:
        nrows = data.shape[0]
    if split_mode is None:
        split_mode = IoConstants.SPM_ANALYSIS_MODES['TRADITIONAL']

    # Sanitize possible out-of-bounds user input
    nrows = min(nrows, data.shape[0])

    if split_mode == IoConstants.SPM_ANALYSIS_MODES['TRADITIONAL']:
        return _split_data_traditional(data, numsets, n1, n2, nrows)
    elif split_mode == IoConstants.SPM_ANALYSIS_MODES['FROZEN_BASELINE']:
        pass
    elif split_mode == IoConstants.SPM_ANALYSIS_MODES['POTENTIATION_CREEP']:
        pass
    else:
        raise ValueError("Unrecognized split_mode ({}) passed to `split_data_for_spm`.".format(split_mode))


def _split_data_traditional(data, numsets, n1, n2, nrows):
    """Called by `split_data_for_spm` for TRADITIONAL SPM analysis.

    Used for SPM analysis comparing time series in group 1 to time series in
    group 2. Splits inputted `data` into:

        Group 1: G1S1, G1S2, G1S3, G1S4, etc.
        Group 2: G2S1, G2S2, G2S3, G2S4, etc.

    Parameters
    ----------
    See `split_data_for_spm`.

    Returns
    -------
    data_tuple : tuple
        Tuple holding group 1 and group 2 series, as for `split_data_for_spm`.

    """
    idxs1 = []
    idxs2 = []
    n = n1 + n2
    for s in range(numsets):
        idxs1.extend(range(s*n, s*n + n1))
        idxs2.extend(range(s*n + n1, (s + 1)*n))
    return _equalize_columns(data[:nrows, idxs1], data[:nrows, idxs2])


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
