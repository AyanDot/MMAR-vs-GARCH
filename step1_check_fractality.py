"""
Step 1: Check if Fractality Exists
Implements partition function analysis to detect moment scaling behavior

Based on paper equation (4.1):
S_q(T, Δt) = Σ |ln(P(iΔt + Δt) / P(iΔt))|^q

CRITICAL: Uses NON-OVERLAPPING intervals as per Zhang's methodology.

If log₁₀(S_q) vs log₁₀(Δt) is linear, then moment scaling exists → fractality confirmed
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import config
from data_loader import DataLoader


class FractalityChecker:
    """
    Checks for multifractal behavior using partition function analysis.

    The partition function S_q(T, Δt) measures how returns scale across different
    time intervals (Δt) and moment orders (q).

    IMPORTANT: Uses NON-OVERLAPPING intervals to avoid artificial correlation.

    If the log-log plot of S_q vs Δt is linear, this indicates moment scaling,
    which is a hallmark of multifractal processes.
    """

    def __init__(self, returns, delta_t_values=None, q_values=None, verbose=config.VERBOSE):
        """
        Initialize fractality checker.

        Parameters:
        -----------
        returns : np.ndarray
            Array of log returns (must be evenly spaced in time)
        delta_t_values : np.ndarray, optional
            Array of Δt values (in number of observations, NOT seconds)
            If None, uses config.generate_delta_t_values()
        q_values : np.ndarray, optional
            Array of q moment orders
            If None, uses config.generate_q_values()
        verbose : bool
            Print detailed information
        """
        self.returns = returns
        self.verbose = verbose

        # Validate returns data
        self._validate_returns()

        # Delta_t values are NOW in number of observations (not seconds)
        if delta_t_values is None:
            self.delta_t_values = config.generate_delta_t_values()
        else:
            self.delta_t_values = delta_t_values

        if q_values is None:
            self.q_values = config.generate_q_values()
        else:
            self.q_values = q_values

        # Storage for results
        self.partition_values = {}  # {q: {delta_t: S_q value}}
        self.r_squared_values = {}  # {q: R²}
        self.slopes = {}  # {q: slope (which equals τ(q) + 1)}
        self.intercepts = {}  # {q: intercept}

        if self.verbose:
            print(f"\nFractality Checker Initialized")
            print(f"Number of returns: {len(returns)}")
            print(f"Δt range: {self.delta_t_values.min()} to {self.delta_t_values.max()} observations")
            print(f"Number of Δt values: {len(self.delta_t_values)}")
            print(f"q range: {self.q_values.min():.2f} to {self.q_values.max():.2f}")
            print(f"Number of q values: {len(self.q_values)}")

    def _validate_returns(self):
        """
        Validate returns data quality.

        Returns:
        --------
        dict
            Dictionary with validation results
        """
        issues = []

        # Check for NaN/Inf
        if np.any(np.isnan(self.returns)):
            issues.append("Contains NaN values")
        if np.any(np.isinf(self.returns)):
            issues.append("Contains infinite values")

        # Check data length
        if len(self.returns) < 1000:
            issues.append(f"Short series ({len(self.returns)} points). Recommended: >10,000")

        # Check for zeros (which could indicate data issues)
        zero_pct = 100 * np.sum(self.returns == 0) / len(self.returns)
        if zero_pct > 5:
            issues.append(f"High percentage of zero returns: {zero_pct:.1f}%")

        if issues:
            print("\n⚠️  DATA QUALITY WARNINGS:")
            for issue in issues:
                print(f"  - {issue}")

            if np.any(np.isnan(self.returns)) or np.any(np.isinf(self.returns)):
                raise ValueError("Cannot proceed with NaN or Inf values in returns")

    def calculate_partition_function(self, q, delta_t):
        """
        Calculate partition function S_q(T, Δt) for given q and Δt.

        *** CRITICAL FIX ***:
        Uses NON-OVERLAPPING intervals as per Zhang's methodology.

        Equation (4.1) from paper:
        S_q(T, Δt) = Σ |ln(P(iΔt + Δt) / P(iΔt))|^q
                   = Σ |r(iΔt, Δt)|^q

        where r(iΔt, Δt) is the return over interval Δt starting at iΔt.

        Zhang explicitly uses N = number of NON-overlapping complete intervals.

        Parameters:
        -----------
        q : float
            Moment order
        delta_t : int
            Time interval (number of observations)

        Returns:
        --------
        float
            S_q value
        """
        # Number of complete NON-overlapping intervals (floor division)
        N = len(self.returns) // delta_t

        if N <= 0:
            return np.nan

        # EFFICIENT VECTORIZED IMPLEMENTATION:
        # Trim returns to fit complete intervals
        trimmed_returns = self.returns[:N * delta_t]

        # Reshape into (N, delta_t) and sum along axis 1
        # This gives us N non-overlapping interval returns
        interval_returns = trimmed_returns.reshape(N, delta_t).sum(axis=1)

        # Calculate partition function: MEAN of |r|^q
        # CRITICAL FIX: Use mean instead of sum to remove N-dependency
        # As Δt increases, N decreases, so sum would confound the scaling
        S_q = np.mean(np.abs(interval_returns) ** q)

        return S_q

    def compute_all_partitions(self):
        """
        Compute partition function for all combinations of q and Δt.

        This is the main computational step that can take some time
        for large datasets or many q/Δt values.
        """
        if self.verbose:
            print("\nComputing partition functions...")
            total_combos = len(self.q_values) * len(self.delta_t_values)
            print(f"Total combinations: {len(self.q_values)} × {len(self.delta_t_values)} = {total_combos}")

        for i, q in enumerate(self.q_values):
            if self.verbose and i % max(1, len(self.q_values) // 10) == 0:
                progress = 100 * i / len(self.q_values)
                print(f"Progress: {progress:.1f}% (q = {q:.2f})")

            self.partition_values[q] = {}

            for delta_t in self.delta_t_values:
                S_q = self.calculate_partition_function(q, delta_t)
                self.partition_values[q][delta_t] = S_q

        if self.verbose:
            print("✓ Partition functions computed")

    def fit_partition_plots(self):
        """
        Fit OLS regression to log-log partition plots.

        For each q, fit: log₁₀(S_q) = slope × log₁₀(Δt) + intercept

        The slope equals τ(q) + 1, so τ(q) = slope - 1
        The R² value indicates how linear the relationship is (evidence of scaling).
        """
        if self.verbose:
            print("\nFitting partition plots...")

        for q in self.q_values:
            # Get delta_t and S_q values
            delta_t_list = []
            S_q_list = []

            for delta_t in sorted(self.partition_values[q].keys()):
                S_q = self.partition_values[q][delta_t]
                if not np.isnan(S_q) and S_q > 0:  # Need positive values for log
                    delta_t_list.append(delta_t)
                    S_q_list.append(S_q)

            if len(delta_t_list) < 2:
                continue

            # Convert to log scale
            log_delta_t = np.log10(delta_t_list)
            log_S_q = np.log10(S_q_list)

            # OLS regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_delta_t, log_S_q)

            # Store results
            # slope = τ(q) + 1  →  τ(q) = slope - 1
            self.slopes[q] = slope
            self.intercepts[q] = intercept
            self.r_squared_values[q] = r_value ** 2

        if self.verbose:
            mean_r2 = np.mean(list(self.r_squared_values.values()))
            min_r2 = np.min(list(self.r_squared_values.values()))
            max_r2 = np.max(list(self.r_squared_values.values()))

            print(f"✓ Partition plots fitted")
            print(f"\nR² Statistics:")
            print(f"  Mean R²: {mean_r2:.4f}")
            print(f"  Min R²: {min_r2:.4f}")
            print(f"  Max R²: {max_r2:.4f}")
            print(f"  Target R² (paper): 0.66")

            if mean_r2 >= config.MIN_R_SQUARED:
                print(f"\n✓ FRACTALITY CONFIRMED (R² = {mean_r2:.4f} >= {config.MIN_R_SQUARED})")
            else:
                print(f"\n✗ Weak evidence of fractality (R² = {mean_r2:.4f} < {config.MIN_R_SQUARED})")

    def get_tau_q(self, q):
        """
        Extract τ(q) from slope.

        From paper: slope = τ(q) + 1
        Therefore: τ(q) = slope - 1

        Parameters:
        -----------
        q : float
            Moment order

        Returns:
        --------
        float or None
            τ(q) value, or None if not available
        """
        if q in self.slopes:
            return self.slopes[q] - 1
        return None

    def detect_crossover_point(self, r_squared_threshold=0.85):
        """
        Detect high-frequency crossover point where scaling breaks down.

        Zhang mentions this is important (Figure 4 in paper).
        Scaling behavior breaks down at very short timescales.

        Parameters:
        -----------
        r_squared_threshold : float
            R² threshold below which we consider scaling has broken down

        Returns:
        --------
        int or None
            Delta_t value where crossover occurs, or None if not detected
        """
        # For each delta_t, check average R² across all q values
        delta_t_r2 = {}

        for delta_t in self.delta_t_values:
            r2_values = []
            for q in self.q_values:
                if q in self.r_squared_values:
                    r2_values.append(self.r_squared_values[q])

            if r2_values:
                delta_t_r2[delta_t] = np.mean(r2_values)

        # Find first delta_t where R² exceeds threshold
        sorted_dt = sorted(delta_t_r2.keys())
        for dt in sorted_dt:
            if delta_t_r2[dt] >= r_squared_threshold:
                if self.verbose:
                    print(f"\nHigh-frequency crossover detected at Δt = {dt} observations")
                    print(f"Scaling behavior valid for Δt >= {dt}")
                return dt

        return None

    def plot_partition_function(self, q_values_to_plot=None, save_path=None):
        """
        Create partition plots (log S_q vs log Δt) for selected q values.

        Parameters:
        -----------
        q_values_to_plot : list, optional
            List of q values to plot. If None, plots a selection of q values.
        save_path : str, optional
            Path to save the plot. If None, displays the plot.
        """
        if q_values_to_plot is None:
            # Select a subset of q values for clearer visualization
            q_values_to_plot = [1, 2, 3, 4, 5]

        plt.figure(figsize=(12, 8))

        colors = plt.cm.viridis(np.linspace(0, 1, len(q_values_to_plot)))

        for idx, q in enumerate(q_values_to_plot):
            if q not in self.partition_values:
                continue

            # Get data points
            delta_t_list = []
            S_q_list = []

            for delta_t in sorted(self.partition_values[q].keys()):
                S_q = self.partition_values[q][delta_t]
                if not np.isnan(S_q) and S_q > 0:
                    delta_t_list.append(delta_t)
                    S_q_list.append(S_q)

            if len(delta_t_list) == 0:
                continue

            # Convert to log scale
            log_delta_t = np.log10(delta_t_list)
            log_S_q = np.log10(S_q_list)

            # Plot data points and regression line
            if q in self.slopes:
                r2 = self.r_squared_values[q]
                plt.plot(log_delta_t, log_S_q, 'o-',
                        color=colors[idx],
                        label=f'q = {q} (R² = {r2:.3f})',
                        alpha=0.7, markersize=4)

                # Plot regression line
                fitted_line = self.slopes[q] * log_delta_t + self.intercepts[q]
                plt.plot(log_delta_t, fitted_line, '--',
                        color=colors[idx], alpha=0.5, linewidth=1)

        plt.xlabel('log₁₀(Δt) [number of observations]', fontsize=12)
        plt.ylabel('log₁₀(S_q)', fontsize=12)
        plt.title('Partition Function: Evidence of Moment Scaling\n(Linear relationship indicates multifractality)',
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)

        # Add R² information
        if len(self.r_squared_values) > 0:
            mean_r2 = np.mean(list(self.r_squared_values.values()))
            plt.text(0.02, 0.98, f'Mean R² = {mean_r2:.4f}\nThreshold = {config.MIN_R_SQUARED}',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_r_squared_vs_q(self, save_path=None):
        """
        Plot R² values as a function of q.

        This shows how well the linear scaling holds across different moment orders.
        """
        if len(self.r_squared_values) == 0:
            print("No R² values available. Run fit_partition_plots() first.")
            return

        q_vals = sorted(self.r_squared_values.keys())
        r2_vals = [self.r_squared_values[q] for q in q_vals]

        plt.figure(figsize=(10, 6))
        plt.plot(q_vals, r2_vals, 'o-', linewidth=2, markersize=6, color='steelblue')
        plt.axhline(y=config.MIN_R_SQUARED, color='r', linestyle='--',
                   linewidth=2, label=f'Threshold = {config.MIN_R_SQUARED}')
        plt.xlabel('q (moment order)', fontsize=12)
        plt.ylabel('R² (goodness of fit)', fontsize=12)
        plt.title('Linearity of Scaling Relationship across Moment Orders', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])

        mean_r2 = np.mean(r2_vals)
        plt.text(0.98, 0.02, f'Mean R² = {mean_r2:.4f}',
                transform=plt.gca().transAxes,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame for further analysis.

        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: q, slope, tau_q, intercept, r_squared
        """
        results = []
        for q in sorted(self.slopes.keys()):
            results.append({
                'q': q,
                'slope': self.slopes[q],
                'tau_q': self.slopes[q] - 1,  # τ(q) = slope - 1
                'intercept': self.intercepts[q],
                'r_squared': self.r_squared_values[q]
            })

        return pd.DataFrame(results)

    def save_results(self, filepath):
        """
        Save results to CSV file.

        Parameters:
        -----------
        filepath : str
            Path to save CSV file
        """
        df = self.get_results_dataframe()
        df.to_csv(filepath, index=False)

        if self.verbose:
            print(f"Results saved to: {filepath}")

    def is_fractal(self, threshold=None):
        """
        Determine if data exhibits fractal behavior.

        Parameters:
        -----------
        threshold : float, optional
            R² threshold. If None, uses config.MIN_R_SQUARED

        Returns:
        --------
        bool
            True if mean R² exceeds threshold
        """
        if threshold is None:
            threshold = config.MIN_R_SQUARED

        if len(self.r_squared_values) == 0:
            return False

        mean_r2 = np.mean(list(self.r_squared_values.values()))
        return mean_r2 >= threshold


def run_fractality_check(returns, output_dir=None, save_plots=True):
    """
    Complete workflow for checking fractality.

    Parameters:
    -----------
    returns : np.ndarray
        Array of log returns
    output_dir : str, optional
        Directory to save results
    save_plots : bool
        Whether to save plots

    Returns:
    --------
    FractalityChecker
        Checker object with results
    """
    # Create output directory
    if output_dir is None:
        output_dir = config.PLOT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize checker
    checker = FractalityChecker(returns)

    # Run analysis
    print("\n" + "="*60)
    print("STEP 1: CHECKING FOR FRACTALITY")
    print("="*60)

    checker.compute_all_partitions()
    checker.fit_partition_plots()

    # Detect crossover point
    checker.detect_crossover_point()

    # Save results
    if config.SAVE_INTERMEDIATE:
        results_path = Path(output_dir) / "step1_partition_results.csv"
        checker.save_results(results_path)

    # Create plots
    if save_plots:
        partition_plot_path = Path(output_dir) / "step1_partition_plots.png"
        checker.plot_partition_function(q_values_to_plot=[1, 2, 3, 4, 5],
                                       save_path=partition_plot_path)

        r2_plot_path = Path(output_dir) / "step1_r_squared.png"
        checker.plot_r_squared_vs_q(save_path=r2_plot_path)

    # Print conclusion
    print("\n" + "="*60)
    if checker.is_fractal():
        print("✓ CONCLUSION: Data exhibits multifractal behavior")
        print("  → Proceed to Step 2: Extract scaling function τ(q)")
    else:
        print("✗ CONCLUSION: Weak evidence of multifractal behavior")
        print("  → Consider using different data or parameters")
    print("="*60 + "\n")

    return checker


if __name__ == "__main__":
    print("Step 1: Check Fractality")
    print("="*60)
    print("\nThis script checks if your price data exhibits multifractal behavior.")
    print("You need to provide returns data to run the analysis.\n")

    # Demo with synthetic multifractal data
    print("Generating synthetic multifractal data for demonstration...")

    # Simple synthetic data (not truly multifractal, just for demo)
    np.random.seed(42)
    n_points = 10000
    returns = np.random.normal(0, 0.001, n_points) * (1 + 0.5 * np.sin(np.arange(n_points) / 100))

    print(f"Generated {n_points} synthetic returns\n")

    # Run analysis
    checker = run_fractality_check(returns, save_plots=True)

    print("\nTo use with real data:")
    print("1. Load your data with DataLoader")
    print("2. Extract returns array")
    print("3. Call run_fractality_check(returns)")
