from pathlib import Path
from collections import namedtuple

class DataConstants:
    DATA_DIR = Path("../../data")
    SAMPLE_DATA = {
            'xlsx': {
                'sample': DATA_DIR.joinpath("mm.xlsx"),
                },
            'csv': {
                'triangle': DATA_DIR.joinpath("triangle.csv"),
                'tmg': DATA_DIR.joinpath("tmg.csv"),
                'spm': DATA_DIR.joinpath("spm.csv"),
                }
            }

class IoConstants:
    TMG_EXCEL_MAGIC_VALUES = {
            # Index of first row with TMG data
            'data_start_row_idx': 24,
            # Index of first column with TMG data
            'data_start_col_idx': 1,
            # Data points in a TMG measurement
            'data_nrows': 1000,
            }

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
    TmgParams = namedtuple('TmgParams', ['dm', 'td', 'tc', 'ts', 'tr'])
    ExtremumParams = namedtuple('ExtremumParams', ['max_time', 'max', 'min_time', 'min'])
    SpmTStatistic = namedtuple('SpmTStatistic', ['t_statistic', 'spm_t'])
    SpmTInference = namedtuple('SpmTInference', ['alpha', 'p', 'threshold', 'clusters'])
    SpmCluster = namedtuple('SpmCluster', ['p', 'start_time', 'end_time', 'centroid_time', 'centroid', 'extremum_time', 'extremum', 'area'])
