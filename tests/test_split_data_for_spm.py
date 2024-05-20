import numpy as np
from tmgtoolkit import tmgio
from tmgtoolkit.constants import IoConstants

def _get_sample_data(nrows, n1, n2, numsets):
    """Constructs sample data for testing SPM split.

    Constructs a 2D Numpy array with well-defined values and structure to use
    for testing the SPM split functions in `tmgio`.

    Returns
    -------
    data : ndarray
        2D Numpy array of shape `(nrows, numsets*(n1 + n2))` with values such that:
        - All group 1 measurements are negative
        - All group 2 measurements are positive
        - Absolute of data point in set N (1-based indexing) is N.

    """
    data = np.zeros((nrows, numsets*(n1 + n2)))
    n = n1 + n2
    for s in range(numsets):
        data[:, s*n:s*n + n1] = -(s + 1)  # group 1
        data[:, s*n + n1:(s+1)*n] = s + 1  # group 2
    return data


def test_split_data_parallel():
    """
    Tests tmgio.split_data_for_spm for split mode `parallel`.
    """
    n1 = 7
    n2 = 8
    numsets = 4
    nrows = 100
    data = _get_sample_data(nrows, n1, n2, numsets)
    assert data.shape == (nrows, (n1 + n2) * numsets)

    split_mode = IoConstants.SPM_SPLIT_MODES['parallel']
    equalize_columns = False
    data_tuples = tmgio.split_data_for_spm(data, numsets, n1, n2, split_mode, equalize_columns=equalize_columns)

    assert len(data_tuples) == numsets

    for s in range(numsets):
        (group1, group2) = data_tuples[s]
        assert len(group1.shape) == 2
        assert len(group2.shape) == 2
        assert group1.shape[0] == nrows
        assert group2.shape[0] == nrows
        assert group1.shape[1] == n1
        assert group2.shape[1] == n2

        # Test data returned by `_get_sample_data` was split correctly.
        # I only sample first row of each measurement because all rows are equal.
        for i in range(n1):
            assert group1[0, i] == -(s + 1)
        for i in range(n2):
            assert group2[0, i] == s + 1


def test_split_data_parallel_all():
    """
    Tests tmgio.split_data_for_spm for split mode `parallel_all`.
    """
    n1 = 7
    n2 = 8
    numsets = 4
    nrows = 100
    data = _get_sample_data(nrows, n1, n2, numsets)
    assert data.shape == (nrows, (n1 + n2) * numsets)

    split_mode = IoConstants.SPM_SPLIT_MODES['parallel_all']
    equalize_columns = False
    group1, group2 = tmgio.split_data_for_spm(data, numsets, n1, n2, split_mode, equalize_columns=equalize_columns)

    assert len(group1.shape) == 2
    assert len(group2.shape) == 2
    assert group1.shape[0] == nrows
    assert group2.shape[0] == nrows
    assert group1.shape[1] == numsets * n1
    assert group2.shape[1] == numsets * n2

    # Test data returned by `_get_sample_data` was split correctly.
    # I only sample first row of each measurement because all rows are equal.
    for s in range(numsets):
        for i in range(n1):
            assert group1[0, s*n1 + i] == -(s + 1)
        for i in range(n2):
            assert group2[0, s*n2 + i] == s + 1


def test_split_data_fixed_baseline():
    """
    Tests tmgio.split_data_for_spm for split mode `fixed_baseline`.
    """
    n1 = 7
    n2 = 8
    numsets = 4
    nrows = 100
    data = _get_sample_data(nrows, n1, n2, numsets)
    assert data.shape == (nrows, (n1 + n2) * numsets)

    split_mode = IoConstants.SPM_SPLIT_MODES['fixed_baseline']
    equalize_columns = False
    data_tuples = tmgio.split_data_for_spm(data, numsets, n1, n2, split_mode, equalize_columns=equalize_columns)

    assert len(data_tuples) == numsets

    for s in range(numsets):
        (group1, group2) = data_tuples[s]
        assert len(group1.shape) == 2
        assert len(group2.shape) == 2
        assert group1.shape[0] == nrows
        assert group2.shape[0] == nrows
        assert group1.shape[1] == n1
        assert group2.shape[1] == n2

        # Test data returned by `_get_sample_data` was split correctly.
        # I only sample first row of each measurement because all rows are equal.
        for i in range(n1):
            assert group1[0, i] == -1
        for i in range(n2):
            assert group2[0, i] == s + 1


def test_split_data_fixed_baseline_all():
    """
    Tests tmgio.split_data_for_spm for split mode `fixed_baseline_all`.
    """
    n1 = 7
    n2 = 8
    numsets = 4
    nrows = 100
    data = _get_sample_data(nrows, n1, n2, numsets)
    assert data.shape == (nrows, (n1 + n2) * numsets)

    split_mode = IoConstants.SPM_SPLIT_MODES['fixed_baseline_all']
    equalize_columns = False
    group1, group2 = tmgio.split_data_for_spm(data, numsets, n1, n2, split_mode, equalize_columns=equalize_columns)

    assert len(group1.shape) == 2
    assert len(group2.shape) == 2
    assert group1.shape[0] == nrows
    assert group2.shape[0] == nrows
    assert group1.shape[1] == n1
    assert group2.shape[1] == numsets * n2

    # Test data returned by `_get_sample_data` was split correctly.
    # I only sample first row of each measurement because all rows are equal.
    for i in range(n1):
        assert group1[0, i] == -1
    for s in range(numsets):
        for i in range(n2):
            assert group2[0, s*n2 + i] == s + 1


def test_split_data_potentiation_creep():
    """
    Tests tmgio.split_data_for_spm for split mode `potentiation_creep`.
    """
    n1 = 7
    n2 = 8
    numsets = 4
    nrows = 100
    data = _get_sample_data(nrows, n1, n2, numsets)
    assert data.shape == (nrows, (n1 + n2) * numsets)

    split_mode = IoConstants.SPM_SPLIT_MODES['potentiation_creep']
    equalize_columns = False
    data_tuples = tmgio.split_data_for_spm(data, numsets, n1, n2, split_mode, equalize_columns=equalize_columns)

    assert len(data_tuples) == numsets - 1

    for s in range(numsets - 1):
        (group1, group2) = data_tuples[s]
        assert len(group1.shape) == 2
        assert len(group2.shape) == 2
        assert group1.shape[0] == nrows
        assert group2.shape[0] == nrows
        assert group1.shape[1] == n1
        assert group2.shape[1] == n1

        # Test data returned by `_get_sample_data` was split correctly.
        # I only sample first row of each measurement because all rows are equal.
        for i in range(n1):
            assert group1[0, i] == -1
        for i in range(n1):
            assert group2[0, i] == -(s + 2)


def test_split_data_potentiation_creep_all():
    """
    Tests tmgio.split_data_for_spm for split mode `potentiation_creep_all`.
    """
    n1 = 7
    n2 = 8
    numsets = 4
    nrows = 100
    data = _get_sample_data(nrows, n1, n2, numsets)
    assert data.shape == (nrows, (n1 + n2) * numsets)

    split_mode = IoConstants.SPM_SPLIT_MODES['potentiation_creep_all']
    equalize_columns = False
    group1, group2 = tmgio.split_data_for_spm(data, numsets, n1, n2, split_mode, equalize_columns=equalize_columns)

    assert len(group1.shape) == 2
    assert len(group2.shape) == 2
    assert group1.shape[0] == nrows
    assert group2.shape[0] == nrows
    assert group1.shape[1] == n1
    assert group2.shape[1] == (numsets - 1) * n1

    # Test data returned by `_get_sample_data` was split correctly.
    # I only sample first row of each measurement because all rows are equal.
    for i in range(n1):
        assert group1[0, i] == -1
    for s in range(numsets - 1):
        for i in range(n1):
            assert group2[0, s*n1 + i] == -(s + 2)
