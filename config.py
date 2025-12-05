"""
Configuration file for MMAR Volatility Forecasting Model
Based on: "Volatility Forecasting with the Multifractal Model of Asset Returns" (Zhang, 2017)
"""

import numpy as np
from datetime import datetime

# =============================================================================
# DATA PARAMETERS
# =============================================================================

# Symbol and timeframe
SYMBOL = "EURUSD"
TIMEFRAME_MT5 = "M15"  # MT5 timeframe code: M1, M5, M15, M30, H1, H4, D1
TIMEFRAME_MINUTES = 15  # For internal calculations (1 min = 1)

# Date range for historical data (training period)
START_DATE = "2025-05-15"
END_DATE = "2025-07-01"

# Forecast parameters
FORECAST_DAYS = 25  # Number of days to forecast
FORECAST_INTERVAL_MINUTES = 10  # Forecast interval (10 minutes as per paper)

# MetaTrader 5 Settings
MT5_ENABLED = True  # Set to False to use CSV files only
MT5_LOGIN = None  # Set your MT5 login (None = use current terminal)
MT5_PASSWORD = ""  # Leave empty if already logged in
MT5_SERVER = ""  # Leave empty if already connected

# =============================================================================
# STEP 1: PARTITION FUNCTION PARAMETERS
# =============================================================================

# Delta t range for partition function (in NUMBER OF OBSERVATIONS)
# Zhang used values spaced by factor of 1.1, from ~400s to 9000s
# For 1-minute data: 400s = 7 obs, 9000s = 150 obs
DELTA_T_MIN = 1  # Minimum number of observations per interval
DELTA_T_MAX = 150  # Maximum number of observations per interval
DELTA_T_SPACING_FACTOR = 1.1  # Logarithmic spacing
NUM_DELTA_T_VALUES = 30  # Number of delta_t values to test

# q values for moment scaling analysis
# Paper used q from 0.01 to 30.00
Q_MIN = 0.01
Q_MAX = 30.0
Q_STEP = 0.5  # Step size for q values

# Linearity threshold (R-squared for partition plots)
# Paper achieved average R² of 0.66
MIN_R_SQUARED = 0.60  # Minimum R² to consider scaling behavior valid

# =============================================================================
# STEP 2: SCALING FUNCTION PARAMETERS
# =============================================================================

# Number of discrete q values for scaling function estimation
NUM_Q_VALUES = 60

# =============================================================================
# STEP 3: MULTIFRACTAL SPECTRUM PARAMETERS
# =============================================================================

# Grid search parameters for fitting spectrum to distributions
ALPHA_0_MIN = 0.1
ALPHA_0_MAX = 1.0
ALPHA_0_STEP = 0.01

# For binomial distribution
ALPHA_MIN_RANGE = (0.01, 0.5)
ALPHA_MAX_RANGE = (0.5, 2.0)
ALPHA_GRID_STEP = 0.01

# For gamma distribution
GAMMA_MIN = 0.1
GAMMA_MAX = 10.0
GAMMA_STEP = 0.1

# =============================================================================
# STEP 4-6: CASCADE AND FBM PARAMETERS
# =============================================================================

# Multiplicative cascade parameters
CASCADE_B = 2  # Binary cascade (divide into 2 intervals at each step)
CASCADE_K_MAX = 10  # Maximum cascade depth (2^10 = 1024 intervals)

# FBM generation parameters
FBM_METHOD = "cholesky"  # Method for generating FBM: 'cholesky' or 'fft'

# =============================================================================
# STEP 7: MONTE CARLO SIMULATION PARAMETERS
# =============================================================================

# Number of Monte Carlo simulations
NUM_SIMULATIONS = 10000  # Paper used 10,000

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# OUTPUT PARAMETERS
# =============================================================================

# Directory for saving results
OUTPUT_DIR = "C:\\Users\\ayanm\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\MMAR_Forecast\\results"

# Directory for saving plots
PLOT_DIR = "C:\\Users\\ayanm\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\MMAR_Forecast\\plots"

# Directory for data
DATA_DIR = "C:\\Users\\ayanm\\AppData\\Roaming\\MetaQuotes\\Terminal\\D0E8209F77C8CF37AD8BF550E51FF075\\MQL5\\Scripts\\MMAR_Forecast\\data"

# Verbose output
VERBOSE = True

# Save intermediate results
SAVE_INTERMEDIATE = True

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_q_values():
    """Generate array of q values for partition function."""
    return np.arange(Q_MIN, Q_MAX + Q_STEP, Q_STEP)

def generate_delta_t_values():
    """
    Generate array of delta t values with logarithmic spacing.

    Returns values in NUMBER OF OBSERVATIONS (not seconds or minutes).
    For 1-minute data, delta_t=60 means 60 one-minute returns = 1 hour.

    Returns:
    --------
    np.ndarray
        Array of delta_t values (integers, number of observations)
    """
    delta_t_values = []
    current = DELTA_T_MIN
    while current <= DELTA_T_MAX:
        delta_t_values.append(int(current))
        current *= DELTA_T_SPACING_FACTOR

    # Remove duplicates and sort
    return np.unique(np.array(delta_t_values))

def get_trading_hours_filter():
    """
    Return function to filter trading hours.
    For EURUSD (24h FX market), we use full day without filtering.

    Note: Seasonality filtering has been removed as we test without it first.
    """
    def filter_func(hour, minute):
        # Allow full 24h trading day
        return True
    return filter_func

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration parameters."""
    assert DELTA_T_MIN > 0, "DELTA_T_MIN must be positive"
    assert DELTA_T_MAX > DELTA_T_MIN, "DELTA_T_MAX must be greater than DELTA_T_MIN"
    assert Q_MIN > 0, "Q_MIN must be positive"
    assert Q_MAX > Q_MIN, "Q_MAX must be greater than Q_MIN"
    assert 0 <= MIN_R_SQUARED <= 1, "MIN_R_SQUARED must be between 0 and 1"
    assert NUM_SIMULATIONS > 0, "NUM_SIMULATIONS must be positive"
    assert CASCADE_B >= 2, "CASCADE_B must be at least 2"
    assert CASCADE_K_MAX > 0, "CASCADE_K_MAX must be positive"

    print("✓ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print(f"\nMMar Forecast Configuration")
    print(f"=" * 50)
    print(f"Symbol: {SYMBOL}")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Forecast: {FORECAST_DAYS} days ahead")
    print(f"Monte Carlo simulations: {NUM_SIMULATIONS}")
    print(f"\nΔt range: {DELTA_T_MIN}s to {DELTA_T_MAX}s")
    print(f"q range: {Q_MIN} to {Q_MAX}")
    print(f"Number of q values: {len(generate_q_values())}")
    print(f"Number of Δt values: {len(generate_delta_t_values())}")
