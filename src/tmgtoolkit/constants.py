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
    SPM_SPLIT_MODES = {
            # Compares each set's group 1 measurements to corresponding group 2
            # measurements. Split returns array of tuples
            # `[(group1_1, group2_1), ..., (group1_N, group2_N)]` where N is
            # the number of measurement sets and:
            # - `group1_1` consists of all measurements in G1S1.
            # - `group2_1` consists of all measurements in G2S1.
            # - `group1_N` consists of all measurements in G1SN.
            # - `group2_N` consists of all measurements in G2SN.
            'parallel': 1,

            # Compares all group 1 measurements to all group 2 measurements.
            # Split returns tuple `(group1, group2)`, where:
            # - `group1` consists of all measurements in G1S1, G1S2, ..., G1SN.
            # - `group2` consists of all measurements in G1S1, G1S2, ..., G1SN.
            'parallel_all': 2,

            # Compares the first set's group 1 measurements to group 2
            # measurements in sets 1, 2, ..., N. Split returns array of tuples
            # `[(group1_1, group2_1), ..., (group1_1, group2_N)]` where N is
            # the number of measurement sets and:
            # - `group1_1` consists of all measurements in G1S1.
            # - `group2_1` consists of all measurements in G2S1.
            # - `group2_N` consists of all measurements in G2SN.
            # Note that each tuple's "group 1" is always set 1's measurements.
            # Use: to avoid potentiation creep in baseline measurements of
            # later sets when e.g. insufficient rest period between sets causes
            # potentiation creep.
            'fixed_baseline': 3,

            # Compares first set's group 1 measurements to all group 2
            # measurements. Split returns tuple `(group1, group2)`, where:
            # - `group1` consists of all measurements in G1S1.
            # - `group2` consists of all measurements in G2S1, G2S2, ..., G2SN, etc.
            'fixed_baseline_all': 4,

            # Compares first set's group 1 measurements to *group 1*
            # measurements from sets 2, 3, ..., N. Split returns array of
            # tuples `[(group1_1, group1_2), ..., (group1_1, group1_N)]`, where
            # N is the number of measurement sets and:
            # - `group1_1` consists of all measurements in G1S1.
            # - `group1_2` consists of all measurements in G1S1.
            # - `group1_N` consists of all measurements in G1SN.
            # Note that measurements are taken only from group 1.
            # Use: to detect "potentiation creep" in baseline measurements of
            # later sets, i.e. when baseline measurements in later sets are
            # faster and higher-amplitude relative to first baseline set when
            # e.g. insufficient rest period between sets causes potentiation
            # creep.
            'potentiation_creep': 5,

            # Compares first set's group 1 measurements to all group 1
            # measurements in sets 2, 3, ..., N.
            # Split returns tuple`(group1, group2)`, where:
            # - `group1` consists of all measurements in G1S1.
            # - `group2` consists of all measurements in G1S2, G1S2, ..., G1SN, etc.
            'potentiation_creep_all': 6,
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
        'tm',
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
