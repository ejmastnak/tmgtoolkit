from constants import IoConstants
import pandas as pd

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


def split_data_for_spm(data, numsets, n1, n2):
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

    Returns
    -------
    data_tuple : tuple
        Tuple holding group 1 and group 2 series. Fields are
        0 (group1) : ndarray
            2D Numpy array holding group 1 measurements.
        1 (group2) : ndarray
            2D Numpy array holding group 2 measurements.

    """
    idxs1 = []
    idxs2 = []
    n = n1 + n2
    for s in range(numsets):
        idxs1.extend(range(s*n, s*n + n1))
        idxs2.extend(range(s*n + n1, (s + 1)*n))
    return (data[:, idxs1], data[:, idxs2])
