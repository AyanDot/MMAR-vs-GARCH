"""
Compare MMAR Forecast with Realized Volatility

This script:
1. Loads MMAR forecast from Step 7
2. Loads out-of-sample (forecast period) data
3. Calculates realized volatility
4. Compares forecast vs realized
5. Computes error metrics
6. Creates visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import config
from data_loader import DataLoader


class ForecastComparison:
    """
    Compare MMAR volatility forecast with realized volatility.
    """

    def __init__(self, forecaster, verbose=True):
        """
        Initialize comparison.

        Parameters:
        -----------
        forecaster : MonteCarloForecaster
            Completed forecaster from Step 7
        verbose : bool
            Print detailed information
        """
        self.forecaster = forecaster
        self.verbose = verbose

        # Forecast values
        self.forecast_volatility = forecaster.mean_forecast
        self.forecast_std = forecaster.std_forecast

        # Realized values (to be computed)
        self.realized_volatility = None
        self.realized_returns = None
        self.forecast_period_data = None

        # Error metrics
        self.error = None
        self.percent_error = None
        self.rmse = None

    def load_forecast_period_data(self, start_date=None, end_date=None, days_ahead=None):
        """
        Load data from the forecast period.

        Parameters:
        -----------
        start_date : str
            Start date for forecast period (YYYY-MM-DD)
            If None, uses day after END_DATE from config
        end_date : str
            End date for forecast period (YYYY-MM-DD)
            If None, uses start_date + days_ahead
        days_ahead : int
            Number of days to forecast (default: from config)

        Returns:
        --------
        pd.DataFrame
            Price data for forecast period
        """
        if self.verbose:
            print("\nLoading forecast period data...")

        # Default to period after training data
        if start_date is None:
            # Parse END_DATE from config and add 1 day
            training_end = pd.to_datetime(config.END_DATE)
            start_date = (training_end + timedelta(days=1)).strftime("%Y-%m-%d")

        if days_ahead is None:
            days_ahead = config.FORECAST_DAYS

        if end_date is None:
            end_date = (pd.to_datetime(start_date) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        if self.verbose:
            print(f"  Forecast period: {start_date} to {end_date}")
            print(f"  Duration: {days_ahead} days")

        # Load data using DataLoader
        loader = DataLoader(
            symbol=config.SYMBOL,
            start_date=start_date,
            end_date=end_date,
            verbose=self.verbose
        )

        # Load from MT5
        if config.MT5_ENABLED:
            loader.load_from_mt5()
        else:
            # Load from CSV if MT5 disabled
            csv_path = Path(config.DATA_DIR) / f"{config.SYMBOL}_{start_date}_{end_date}.csv"
            if csv_path.exists():
                loader.load_from_csv(csv_path)
            else:
                raise FileNotFoundError(f"No data found for forecast period. Expected: {csv_path}")

        # Calculate returns
        loader.calculate_returns(price_column='close', method='log')

        self.forecast_period_data = loader.data
        self.realized_returns = loader.get_returns_array()

        if self.verbose:
            print(f"  ✓ Loaded {len(self.realized_returns)} returns")
            print(f"  Return range: [{self.realized_returns.min():.6f}, {self.realized_returns.max():.6f}]")

        return self.forecast_period_data

    def calculate_realized_volatility(self):
        """
        Calculate realized volatility from forecast period data.

        Returns:
        --------
        float
            Realized volatility (standard deviation of returns)
        """
        if self.realized_returns is None:
            raise ValueError("Must load forecast period data first!")

        if self.verbose:
            print("\nCalculating realized volatility...")

        # Standard deviation of returns
        self.realized_volatility = np.std(self.realized_returns, ddof=1)  # Unbiased estimator

        if self.verbose:
            print(f"  ✓ Realized volatility: {self.realized_volatility:.6f}")

        return self.realized_volatility

    def calculate_rolling_realized_volatility(self, window=288):
        """
        Calculate rolling realized volatility.

        Useful for seeing how volatility evolved during forecast period.

        Parameters:
        -----------
        window : int
            Rolling window size (default: 288 = 1 day for 5-min data)

        Returns:
        --------
        pd.Series
            Rolling realized volatility
        """
        if self.realized_returns is None:
            raise ValueError("Must load forecast period data first!")

        returns_series = pd.Series(self.realized_returns)
        rolling_vol = returns_series.rolling(window=window).std()

        return rolling_vol

    def compute_error_metrics(self):
        """
        Compute forecast error metrics.

        Returns:
        --------
        dict
            Dictionary with error metrics
        """
        if self.realized_volatility is None:
            raise ValueError("Must calculate realized volatility first!")

        if self.verbose:
            print("\nComputing error metrics...")

        # Absolute error
        self.error = self.forecast_volatility - self.realized_volatility

        # Percentage error
        self.percent_error = 100 * self.error / self.realized_volatility

        # Root Mean Squared Error (treating forecast as single prediction)
        self.rmse = np.sqrt((self.error)**2)

        # Mean Absolute Percentage Error
        mape = np.abs(self.percent_error)

        # Check if forecast is within 95% CI
        lower_bound = self.forecast_volatility - 1.96 * self.forecast_std
        upper_bound = self.forecast_volatility + 1.96 * self.forecast_std
        within_ci = lower_bound <= self.realized_volatility <= upper_bound

        results = {
            'forecast': self.forecast_volatility,
            'realized': self.realized_volatility,
            'error': self.error,
            'percent_error': self.percent_error,
            'abs_percent_error': mape,
            'rmse': self.rmse,
            'forecast_std': self.forecast_std,
            'forecast_95ci_lower': lower_bound,
            'forecast_95ci_upper': upper_bound,
            'within_95ci': within_ci
        }

        if self.verbose:
            print(f"\n  Forecast:  {self.forecast_volatility:.6f}")
            print(f"  Realized:  {self.realized_volatility:.6f}")
            print(f"  Error:     {self.error:.6f}")
            print(f"  % Error:   {self.percent_error:.2f}%")
            print(f"  MAPE:      {mape:.2f}%")
            print(f"  RMSE:      {self.rmse:.6f}")
            print(f"\n  95% CI: [{lower_bound:.6f}, {upper_bound:.6f}]")
            print(f"  Realized within CI: {'✓ YES' if within_ci else '✗ NO'}")

        return results

    def plot_comparison(self, save_path=None):
        """
        Create comprehensive comparison visualizations.
        """
        if self.realized_volatility is None:
            raise ValueError("Must calculate realized volatility first!")

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Forecast vs Realized (bar chart)
        ax1 = fig.add_subplot(gs[0, 0])
        x = ['MMAR Forecast', 'Realized']
        y = [self.forecast_volatility, self.realized_volatility]
        colors = ['steelblue', 'darkgreen']

        bars = ax1.bar(x, y, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

        # Add error bars for forecast uncertainty
        ax1.errorbar([0], [self.forecast_volatility],
                    yerr=[1.96 * self.forecast_std],
                    fmt='none', color='red', linewidth=2,
                    capsize=10, label='95% CI')

        ax1.set_ylabel('Volatility', fontsize=12)
        ax1.set_title('MMAR Forecast vs Realized Volatility', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, y)):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.00005,
                    f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 2. Returns distribution (forecast period)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.realized_returns, bins=50, color='green', alpha=0.6,
                edgecolor='black', density=True, label='Realized returns')

        # Overlay normal distribution with forecasted volatility
        x_range = np.linspace(self.realized_returns.min(), self.realized_returns.max(), 100)
        from scipy.stats import norm
        forecast_dist = norm.pdf(x_range, 0, self.forecast_volatility)
        realized_dist = norm.pdf(x_range, 0, self.realized_volatility)

        ax2.plot(x_range, forecast_dist, 'b--', linewidth=2, label=f'Forecast N(0, {self.forecast_volatility:.6f})')
        ax2.plot(x_range, realized_dist, 'g-', linewidth=2, label=f'Realized N(0, {self.realized_volatility:.6f})')

        ax2.set_xlabel('Return', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Return Distribution\n(Forecast Period)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 3. Rolling realized volatility
        ax3 = fig.add_subplot(gs[1, :])
        rolling_vol = self.calculate_rolling_realized_volatility(window=288)

        ax3.plot(rolling_vol, 'g-', linewidth=1.5, label='Rolling realized vol (1 day)', alpha=0.8)
        ax3.axhline(self.forecast_volatility, color='blue', linestyle='--',
                   linewidth=2, label='MMAR forecast')
        ax3.axhline(self.realized_volatility, color='darkgreen', linestyle='-',
                   linewidth=2, label='Mean realized vol')

        # Forecast CI
        ax3.fill_between(range(len(rolling_vol)),
                        self.forecast_volatility - 1.96 * self.forecast_std,
                        self.forecast_volatility + 1.96 * self.forecast_std,
                        color='blue', alpha=0.2, label='Forecast 95% CI')

        ax3.set_xlabel('Time (observations)', fontsize=11)
        ax3.set_ylabel('Volatility', fontsize=11)
        ax3.set_title('Rolling Realized Volatility vs MMAR Forecast', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10, loc='best')
        ax3.grid(True, alpha=0.3)

        # 4. Error metrics
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis('off')

        error_text = f"""
        FORECAST PERFORMANCE METRICS
        {'='*45}

        Forecast:           {self.forecast_volatility:.6f}
        Realized:           {self.realized_volatility:.6f}

        Error:              {self.error:.6f}
        % Error:            {self.percent_error:.2f}%
        MAPE:               {abs(self.percent_error):.2f}%
        RMSE:               {self.rmse:.6f}

        Forecast Std Dev:   {self.forecast_std:.6f}
        95% CI:             [{self.forecast_volatility - 1.96*self.forecast_std:.6f},
                             {self.forecast_volatility + 1.96*self.forecast_std:.6f}]

        Realized in CI:     {'YES ✓' if self.compute_error_metrics()['within_95ci'] else 'NO ✗'}

        {'='*45}
        """

        ax4.text(0.1, 0.5, error_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # 5. Annualized comparison
        ax5 = fig.add_subplot(gs[2, 1])

        # Convert to annualized (assuming 5-min data)
        periods_per_day = 288  # 5-min periods in 24h
        periods_per_year = periods_per_day * 252  # Trading days

        ann_forecast = self.forecast_volatility * np.sqrt(periods_per_year) * 100
        ann_realized = self.realized_volatility * np.sqrt(periods_per_year) * 100

        categories = ['5-min', 'Hourly', 'Daily', 'Annual']
        forecast_vals = [
            self.forecast_volatility * 100,
            self.forecast_volatility * np.sqrt(12) * 100,
            self.forecast_volatility * np.sqrt(288) * 100,
            ann_forecast
        ]
        realized_vals = [
            self.realized_volatility * 100,
            self.realized_volatility * np.sqrt(12) * 100,
            self.realized_volatility * np.sqrt(288) * 100,
            ann_realized
        ]

        x_pos = np.arange(len(categories))
        width = 0.35

        ax5.bar(x_pos - width/2, forecast_vals, width, label='MMAR Forecast',
               color='steelblue', alpha=0.7, edgecolor='black')
        ax5.bar(x_pos + width/2, realized_vals, width, label='Realized',
               color='darkgreen', alpha=0.7, edgecolor='black')

        ax5.set_xlabel('Timeframe', fontsize=11)
        ax5.set_ylabel('Volatility (%)', fontsize=11)
        ax5.set_title('Forecast vs Realized\n(Different Timeframes)', fontsize=12, fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(categories)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')

        plt.suptitle('MMAR Forecast Validation', fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"\nPlot saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def run_forecast_comparison(forecaster=None, start_date=None, end_date=None,
                            days_ahead=None, output_dir=None, save_plots=True):
    """
    Complete workflow for comparing MMAR forecast with realized volatility.

    Parameters:
    -----------
    forecaster : MonteCarloForecaster, optional
        Forecaster from Step 7. If None, loads from pickle.
    start_date : str, optional
        Start of forecast period (YYYY-MM-DD)
    end_date : str, optional
        End of forecast period (YYYY-MM-DD)
    days_ahead : int, optional
        Number of days to forecast
    output_dir : str, optional
        Directory to save results
    save_plots : bool
        Whether to save plots

    Returns:
    --------
    ForecastComparison
        Comparison object with results
    """
    # Create output directory
    if output_dir is None:
        output_dir = config.PLOT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load forecaster if not provided
    if forecaster is None:
        forecaster_path = Path(config.OUTPUT_DIR) / "step7_forecaster.pkl"
        if not forecaster_path.exists():
            raise FileNotFoundError(f"Step 7 results not found: {forecaster_path}")

        print(f"Loading forecaster from: {forecaster_path}")
        with open(forecaster_path, 'rb') as f:
            forecaster = pickle.load(f)

    # Initialize comparison
    comparison = ForecastComparison(forecaster)

    # Run analysis
    print("\n" + "="*70)
    print(" "*15 + "MMAR FORECAST VALIDATION")
    print("="*70)

    # Load forecast period data
    comparison.load_forecast_period_data(start_date, end_date, days_ahead)

    # Calculate realized volatility
    comparison.calculate_realized_volatility()

    # Compute error metrics
    results = comparison.compute_error_metrics()

    # Create plots
    if save_plots:
        comparison_plot_path = Path(output_dir) / "forecast_comparison.png"
        comparison.plot_comparison(save_path=comparison_plot_path)

    # Save results
    results_path = Path(output_dir) / "forecast_comparison_results.csv"
    results_df = pd.DataFrame([results])
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\n" + "="*70)
    print("FORECAST VALIDATION COMPLETE")
    print("="*70)

    print(f"\nForecast Quality:")
    if results['within_95ci']:
        print(f"  ✓✓ EXCELLENT: Realized volatility within 95% CI")
    elif abs(results['percent_error']) < 10:
        print(f"  ✓ GOOD: Forecast error < 10%")
    elif abs(results['percent_error']) < 20:
        print(f"  ~ ACCEPTABLE: Forecast error < 20%")
    else:
        print(f"  ✗ POOR: Forecast error > 20%")

    print(f"\nNext steps:")
    print(f"  1. Compare with GARCH benchmark")
    print(f"  2. Test on multiple forecast periods")
    print(f"  3. Analyze forecast errors for patterns")
    print("="*70 + "\n")

    return comparison


if __name__ == "__main__":
    print("MMAR Forecast Validation")
    print("="*70)
    print("\nThis script compares MMAR forecast with realized volatility.")
    print("You must run Step 7 first.\n")

    # Run comparison
    comparison = run_forecast_comparison(save_plots=True)
