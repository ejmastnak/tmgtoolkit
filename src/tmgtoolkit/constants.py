from pathlib import Path
from collections import namedtuple

class IoConstants:
    TMG_EXCEL_MAGIC_VALUES = {
            # Index of first row with TMG data
            'data_start_row_idx': 24,
            # Index of first column with TMG data
            'data_start_col_idx': 1,
            # Data points in a TMG measurement
            'data_nrows': 1000,
            }
    SPM_ANALYSIS_MODES = {
            # Compares measurements in group 1 to measurements in group 2.
            # Group 1: G1S1, G1S2, G1S3, G1S4, etc.
            # Group 2: G2S1, G2S2, G2S3, G2S4, etc.
            'traditional': 1,

            # Compares measurements from first set of group 1 to group 2
            # measurements. Applicable when group 1 represents baseline
            # measurements in a baseline-potentiated measurement protocol. Used
            # to avoid influence of lingering potentiation in baseline
            # measurements of later sets due to e.g. insufficient rest period
            # between conditioning exercise of one set and the baseline
            # measurements of the next set.
            # Group 1: G1S1 measurements only
            # Group 2: G2S1, G2S2, G2S3, G2S4, etc.
            'frozen_baseline': 2,

            # Compares measurements from first set of group 1 to later group 1
            # measurements. A complement to `frozen_baseline`, again used when
            # group 1 represents baseline measurements in a
            # baseline-potentiated measurement protocol. Used to detect
            # "potentiation creep" in baseline measurements of later sets, i.e.
            # if measurements in later baseline sets are faster and
            # higher-amplitude relative to first baseline set.
            # Group 1: G1S1 measurements only
            # Group 2: G1S2, G1S3, G1S4, etc.
            'potentiation_creep': 3,
            }
    # Controls amount of noise added to padded time series when equalizing columns
    NOISE_SCALE = 0.001

class TimeSeriesConstants:
    TMG_PARAMS = {
            # To ignore IIR filter artefact in first few ms of TMG signal
            'ignore_maxima_with_idx_less_than': 8,
            # To ignore IIR filter artefact in first few ms of TMG signal
            'ignore_maxima_less_than': 0.1,  # mm assumed
            # When interpolating to find times of target amplitudes
            'interpolation_window_padding': 1,
            }
    EXTREMUM_PARAMS = {
            # When interpolating to find extrema of time series signals
            'interpolation_window_padding': 2,
            # Point density in polynomial evaluation grid relative to interpolation window
            'poly_grid_magnification': 100,
            }

class SpmConstants:
    T_STATISTIC = {
            # For mitigating influence of IIR filter artefact on SPM statistic
            'iir_artefact_mitigation_rows': 4,
            }
    CLUSTER = {
            # When interpolating to find extrema of t-statistic curve
            'interpolation_window_padding': 1,
            }

class PlottingConstants:
    TIME_SERIES_DEFAULTS = {
            'xlabel': 'Time',
            'ylabel': 'Signal',
            'title': 'Time series plot',
            'marker': '.',
            }
    SPM_STATISTIC_DEFAULTS = {
            'xlabel': 'Time',
            'ylabel': 'Signal',
            'title': 'Time series plot',
            'marker': '.',
            'x_axis_color': '#000000',
            'x_axis_linestyle': ':',
            'threshold_color': '#000000',
            'threshold_linestyle': '--',
            'cluster_fillcolor': '#aaccff',  # light blue
            'textbox_x': 0.88,
            'textbox_y': 0.97,
            'textbox_facecolor': '#ffffff',
            'textbox_edgecolor': '#222222',
            'textbox_style': 'round,pad=0.3',
            }
    SPM_INPUT_DATA_DEFAULTS = {
            'xlabel': 'Time',
            'ylabel': 'Signal',
            'title': 'Time series plot',
            'color1': '#000000',
            'color2': '#0044aa',  # dark blue
            'alpha1': 0.20,
            'alpha2': 0.75,
            'linewidth': 2.0,
            'label1': 'Group 1',
            'label2': 'Group 2',
            'z_line1': 4,
            'z_line2': 3,
            'z_fill1': 1,
            'z_fill2': 2,
            'x_axis_color': '#000000',
            'x_axis_linestyle': ':',
            'legend_alpha': 1.0,
            }

class NamedTupleTypes:
    TmgParams = namedtuple('TmgParams', [
        'dm',
        'td',
        'tc',
        'ts',
        'tr'
        ])
    ExtremumParams = namedtuple('ExtremumParams', [
        'max_time',
        'max',
        'min_time',
        'min'
        ])
    SpmTStatistic = namedtuple('SpmTStatistic', [
        't_statistic',
        'spm_t'
        ])
    SpmTInference = namedtuple('SpmTInference', [
        'alpha',
        'p',
        'threshold',
        'clusters'
        ])
    SpmCluster = namedtuple('SpmCluster', [
        'idx',
        'p',
        'start_time',
        'end_time',
        'centroid_time',
        'centroid',
        'extremum_time',
        'extremum',
        'area'
        ])
