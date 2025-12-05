"""
Utility functions for MMAR Volatility Forecasting
Helper functions used across multiple steps
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def ensure_directory(path):
    """
    Ensure directory exists, create if it doesn't.

    Parameters:
    -----------
    path : str or Path
        Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig, filepath, dpi=300, verbose=True):
    """
    Save matplotlib figure with consistent settings.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filepath : str
        Path to save figure
    dpi : int
        Resolution (default: 300)
    verbose : bool
        Print confirmation message
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    if verbose:
        print(f"Figure saved: {filepath}")
    plt.close(fig)


def calculate_statistics(data):
    """
    Calculate summary statistics for data.

    Parameters:
    -----------
    data : np.ndarray or pd.Series
        Data array

    Returns:
    --------
    dict
        Dictionary of statistics
    """
    return {
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'median': np.median(data),
        'skewness': pd.Series(data).skew(),
        'kurtosis': pd.Series(data).kurtosis(),
        'count': len(data)
    }


def print_statistics(data, name="Data"):
    """
    Print formatted statistics.

    Parameters:
    -----------
    data : np.ndarray or pd.Series
        Data array
    name : str
        Name of the data for printing
    """
    stats = calculate_statistics(data)

    print(f"\n{name} Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']:.8f}")
    print(f"  Std Dev: {stats['std']:.8f}")
    print(f"  Min: {stats['min']:.8f}")
    print(f"  Max: {stats['max']:.8f}")
    print(f"  Median: {stats['median']:.8f}")
    print(f"  Skewness: {stats['skewness']:.4f}")
    print(f"  Kurtosis: {stats['kurtosis']:.4f}")


def format_time_delta(seconds):
    """
    Format time duration in human-readable format.

    Parameters:
    -----------
    seconds : float
        Time in seconds

    Returns:
    --------
    str
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def check_array_valid(arr, name="Array"):
    """
    Check if array contains valid numerical values.

    Parameters:
    -----------
    arr : np.ndarray
        Array to check
    name : str
        Name for error messages

    Raises:
    -------
    ValueError
        If array contains NaN or infinite values
    """
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(arr)):
        raise ValueError(f"{name} contains infinite values")


def rolling_window(arr, window_size):
    """
    Create rolling windows from array.

    Parameters:
    -----------
    arr : np.ndarray
        Input array
    window_size : int
        Size of rolling window

    Returns:
    --------
    np.ndarray
        Array of shape (n_windows, window_size)
    """
    n = len(arr) - window_size + 1
    return np.array([arr[i:i+window_size] for i in range(n)])


def aggregate_returns(returns, aggregation_factor):
    """
    Aggregate returns over longer intervals.

    For example, aggregate 1-minute returns into 5-minute returns.

    Parameters:
    -----------
    returns : np.ndarray
        Array of log returns
    aggregation_factor : int
        Number of periods to aggregate

    Returns:
    --------
    np.ndarray
        Aggregated returns
    """
    n_complete = (len(returns) // aggregation_factor) * aggregation_factor
    truncated = returns[:n_complete]
    reshaped = truncated.reshape(-1, aggregation_factor)
    aggregated = np.sum(reshaped, axis=1)
    return aggregated


def calculate_volatility(returns, window_size=None):
    """
    Calculate volatility (standard deviation) of returns.

    Parameters:
    -----------
    returns : np.ndarray
        Array of returns
    window_size : int, optional
        If provided, calculates rolling volatility

    Returns:
    --------
    float or np.ndarray
        Volatility value or rolling volatility array
    """
    if window_size is None:
        return np.std(returns)
    else:
        windows = rolling_window(returns, window_size)
        return np.std(windows, axis=1)


def log_progress(current, total, step=10):
    """
    Print progress at regular intervals.

    Parameters:
    -----------
    current : int
        Current iteration
    total : int
        Total iterations
    step : int
        Print every 'step' percent
    """
    percent = 100 * current / total
    if current % max(1, total // (100 // step)) == 0:
        print(f"Progress: {percent:.1f}%")


class Timer:
    """Simple timer context manager."""

    def __init__(self, name="Operation", verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        if self.verbose:
            print(f"\n{self.name} started...")
        self.start_time = pd.Timestamp.now()
        return self

    def __exit__(self, *args):
        self.end_time = pd.Timestamp.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        if self.verbose:
            print(f"{self.name} completed in {format_time_delta(elapsed)}")

    @property
    def elapsed(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


def normalize_array(arr):
    """
    Normalize array to zero mean and unit variance.

    Parameters:
    -----------
    arr : np.ndarray
        Input array

    Returns:
    --------
    np.ndarray
        Normalized array
    """
    return (arr - np.mean(arr)) / np.std(arr)


def standardize_returns(returns):
    """
    Standardize returns to have zero mean and unit variance.

    Parameters:
    -----------
    returns : np.ndarray
        Array of returns

    Returns:
    --------
    tuple
        (standardized_returns, mean, std)
    """
    mean = np.mean(returns)
    std = np.std(returns)
    standardized = (returns - mean) / std
    return standardized, mean, std


def unstandardize_returns(standardized_returns, mean, std):
    """
    Reverse standardization.

    Parameters:
    -----------
    standardized_returns : np.ndarray
        Standardized returns
    mean : float
        Original mean
    std : float
        Original std dev

    Returns:
    --------
    np.ndarray
        Original scale returns
    """
    return standardized_returns * std + mean


def calculate_autocorrelation(data, max_lag=50):
    """
    Calculate autocorrelation function.

    Parameters:
    -----------
    data : np.ndarray
        Time series data
    max_lag : int
        Maximum lag to calculate

    Returns:
    --------
    np.ndarray
        Autocorrelation values for lags 0 to max_lag
    """
    mean = np.mean(data)
    var = np.var(data)
    n = len(data)

    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0

    for lag in range(1, max_lag + 1):
        c = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / n
        acf[lag] = c / var

    return acf


def plot_autocorrelation(data, max_lag=50, title="Autocorrelation", save_path=None):
    """
    Plot autocorrelation function.

    Parameters:
    -----------
    data : np.ndarray
        Time series data
    max_lag : int
        Maximum lag
    title : str
        Plot title
    save_path : str, optional
        Path to save plot
    """
    acf = calculate_autocorrelation(data, max_lag)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stem(range(len(acf)), acf, basefmt=" ")
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    # Add confidence intervals (95%)
    conf_interval = 1.96 / np.sqrt(len(data))
    ax.axhline(y=conf_interval, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-conf_interval, color='r', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()


def create_summary_report(results_dict, title="Analysis Summary"):
    """
    Create formatted summary report.

    Parameters:
    -----------
    results_dict : dict
        Dictionary of results to report
    title : str
        Report title

    Returns:
    --------
    str
        Formatted report string
    """
    report = []
    report.append("\n" + "="*60)
    report.append(title)
    report.append("="*60)

    for key, value in results_dict.items():
        if isinstance(value, (int, np.integer)):
            report.append(f"{key}: {value}")
        elif isinstance(value, (float, np.floating)):
            report.append(f"{key}: {value:.6f}")
        else:
            report.append(f"{key}: {value}")

    report.append("="*60 + "\n")

    return "\n".join(report)


def save_results_to_json(results_dict, filepath):
    """
    Save results dictionary to JSON file.

    Parameters:
    -----------
    results_dict : dict
        Results to save
    filepath : str
        Path to JSON file
    """
    import json

    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    converted = convert_types(results_dict)

    with open(filepath, 'w') as f:
        json.dump(converted, f, indent=2)

    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    print("MMAR Utils - Utility Functions")
    print("="*60)
    print("\nAvailable utility functions:")
    print("  - ensure_directory(path)")
    print("  - save_figure(fig, filepath)")
    print("  - calculate_statistics(data)")
    print("  - print_statistics(data)")
    print("  - check_array_valid(arr)")
    print("  - rolling_window(arr, window_size)")
    print("  - aggregate_returns(returns, factor)")
    print("  - calculate_volatility(returns)")
    print("  - Timer context manager")
    print("  - normalize_array(arr)")
    print("  - calculate_autocorrelation(data)")
    print("  - plot_autocorrelation(data)")
    print("  - create_summary_report(results_dict)")
    print("  - save_results_to_json(results_dict, filepath)")
