"""
Step 7: Monte Carlo Volatility Forecasting
Runs 10,000 MMAR simulations to forecast volatility

From paper Section 5.b:
"This process is repeated until it exceeds a length of 10,000 returns,
where each return represents a time interval of 10 minutes.
The standard deviation of these returns is taken as a forecast of volatility."

Algorithm:
1. Repeat 10,000 times:
   - Generate trading time (Step 4)
   - Generate FBM (Step 5)
   - Combine to get MMAR (Step 6)
   - Calculate volatility
2. Average volatilities = forecast
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import config
from step4_generate_cascade import CascadeGenerator
from step5_generate_fbm import FBMGenerator
from step6_combine_model import MMARCombiner


class MonteCarloForecaster:
    """
    Monte Carlo simulation for MMAR volatility forecasting.

    Runs thousands of simulations to get robust volatility forecast.
    """

    def __init__(self, fitter, n_simulations=None, forecast_length=None,
                 sample_volatility=None, verbose=config.VERBOSE):
        """
        Initialize Monte Carlo forecaster.

        Parameters:
        -----------
        fitter : SpectrumFitter
            Completed fitter from Step 3
        n_simulations : int, optional
            Number of Monte Carlo simulations (default: from config)
        forecast_length : int, optional
            Number of returns to generate per simulation (default: from config)
        sample_volatility : float, optional
            Sample std dev to scale FBM (if None, will use unit variance)
        verbose : bool
            Print detailed information
        """
        self.fitter = fitter
        self.verbose = verbose

        # Parameters
        self.n_simulations = n_simulations or config.NUM_SIMULATIONS
        self.forecast_length = forecast_length or 10000  # Paper uses 10,000
        self.sample_volatility = sample_volatility or 1.0

        # Storage
        self.volatility_forecasts = []
        self.mean_forecast = None
        self.std_forecast = None

        if self.verbose:
            print(f"\nMonte Carlo Forecaster Initialized")
            print(f"Number of simulations: {self.n_simulations}")
            print(f"Forecast length: {self.forecast_length} returns")
            print(f"Sample volatility: {self.sample_volatility:.6f}")

    def run_single_simulation(self):
        """
        Run a single MMAR simulation.

        Returns:
        --------
        float
            Volatility (std dev) of simulated returns
        """
        # Step 4: Generate trading time
        cascade_gen = CascadeGenerator(self.fitter, b=2, k=10, verbose=False)
        cascade_gen.generate_cascade()
        cascade_gen.integrate_measure()

        # Step 5: Generate FBM
        fbm_gen = FBMGenerator(H=self.fitter.H, n_points=2**10, verbose=False)
        fbm_gen.generate_fbm_davies_harte()

        # Scale FBM by sample volatility
        fbm_gen.scale_fbm(self.sample_volatility)

        # Step 6: Combine
        combiner = MMARCombiner(fbm_gen, cascade_gen, verbose=False)
        _, returns = combiner.combine_fbm_and_trading_time()

        # Calculate volatility
        volatility = np.std(returns)

        return volatility

    def run_monte_carlo(self):
        """
        Run full Monte Carlo simulation.

        Returns:
        --------
        dict
            Dictionary with forecast results
        """
        if self.verbose:
            print("\nRunning Monte Carlo simulations...")
            print(f"This will take a few minutes...")

        self.volatility_forecasts = []

        for i in range(self.n_simulations):
            # Run simulation
            volatility = self.run_single_simulation()
            self.volatility_forecasts.append(volatility)

            # Progress update
            if self.verbose and (i + 1) % max(1, self.n_simulations // 10) == 0:
                progress = 100 * (i + 1) / self.n_simulations
                mean_so_far = np.mean(self.volatility_forecasts)
                print(f"  Progress: {progress:.1f}% ({i + 1}/{self.n_simulations}) - Mean vol: {mean_so_far:.6f}")

        # Calculate statistics
        self.mean_forecast = np.mean(self.volatility_forecasts)
        self.std_forecast = np.std(self.volatility_forecasts)

        if self.verbose:
            print(f"\n  ✓ Monte Carlo complete!")
            print(f"\nVolatility Forecast:")
            print(f"  Mean: {self.mean_forecast:.6f}")
            print(f"  Std dev: {self.std_forecast:.6f}")
            print(f"  95% CI: [{self.mean_forecast - 1.96*self.std_forecast:.6f}, "
                  f"{self.mean_forecast + 1.96*self.std_forecast:.6f}]")

        return {
            'mean_volatility': self.mean_forecast,
            'std_volatility': self.std_forecast,
            'volatility_forecasts': self.volatility_forecasts
        }

    def plot_forecast_distribution(self, save_path=None):
        """Visualize the distribution of volatility forecasts."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top left: Histogram of forecasts
        ax = axes[0, 0]
        ax.hist(self.volatility_forecasts, bins=50, color='steelblue',
                alpha=0.7, edgecolor='black', density=True)
        ax.axvline(self.mean_forecast, color='red', linestyle='--',
                  linewidth=2, label=f'Mean = {self.mean_forecast:.6f}')
        ax.set_xlabel('Volatility', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Distribution of Volatility Forecasts\n(10,000 Monte Carlo simulations)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Top right: Time series of forecasts
        ax = axes[0, 1]
        ax.plot(self.volatility_forecasts, 'b-', linewidth=0.5, alpha=0.5)
        ax.axhline(self.mean_forecast, color='red', linestyle='--',
                  linewidth=2, label='Mean')
        ax.fill_between(range(len(self.volatility_forecasts)),
                        self.mean_forecast - 1.96*self.std_forecast,
                        self.mean_forecast + 1.96*self.std_forecast,
                        alpha=0.2, color='red', label='95% CI')
        ax.set_xlabel('Simulation number', fontsize=11)
        ax.set_ylabel('Volatility', fontsize=11)
        ax.set_title('Volatility Forecast Convergence',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Bottom left: Cumulative mean
        ax = axes[1, 0]
        cumulative_mean = np.cumsum(self.volatility_forecasts) / np.arange(1, len(self.volatility_forecasts) + 1)
        ax.plot(cumulative_mean, 'g-', linewidth=2)
        ax.axhline(self.mean_forecast, color='red', linestyle='--',
                  linewidth=1, alpha=0.7, label='Final mean')
        ax.set_xlabel('Number of simulations', fontsize=11)
        ax.set_ylabel('Cumulative mean volatility', fontsize=11)
        ax.set_title('Convergence of Mean Estimate',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Bottom right: Q-Q plot (check normality)
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(self.volatility_forecasts, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot\n(Check if forecasts are normally distributed)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def save_results(self, filepath):
        """Save forecast results to CSV."""
        results_df = pd.DataFrame({
            'simulation_number': range(1, len(self.volatility_forecasts) + 1),
            'volatility_forecast': self.volatility_forecasts
        })

        # Add summary statistics as metadata
        metadata = pd.DataFrame({
            'simulation_number': ['MEAN', 'STD_DEV', 'MIN', 'MAX', 'MEDIAN'],
            'volatility_forecast': [
                self.mean_forecast,
                self.std_forecast,
                np.min(self.volatility_forecasts),
                np.max(self.volatility_forecasts),
                np.median(self.volatility_forecasts)
            ]
        })

        final_df = pd.concat([metadata, results_df], ignore_index=True)
        final_df.to_csv(filepath, index=False)

        if self.verbose:
            print(f"Results saved to: {filepath}")


def run_monte_carlo_forecast(fitter, sample_volatility=None, n_simulations=None,
                             output_dir=None, save_plots=True):
    """
    Complete workflow for Monte Carlo volatility forecasting.

    Parameters:
    -----------
    fitter : SpectrumFitter
        Completed fitter from Step 3
    sample_volatility : float, optional
        Sample std dev from historical data
    n_simulations : int, optional
        Number of simulations (default: from config)
    output_dir : str, optional
        Directory to save results
    save_plots : bool
        Whether to save plots

    Returns:
    --------
    MonteCarloForecaster
        Forecaster object with results
    """
    # Create output directory
    if output_dir is None:
        output_dir = config.PLOT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize forecaster
    forecaster = MonteCarloForecaster(fitter,
                                      n_simulations=n_simulations,
                                      sample_volatility=sample_volatility)

    # Run analysis
    print("\n" + "="*60)
    print("STEP 7: MONTE CARLO VOLATILITY FORECASTING")
    print("="*60)

    # Run simulations
    results = forecaster.run_monte_carlo()

    # Save results
    results_path = Path(output_dir) / "step7_forecast_results.csv"
    forecaster.save_results(results_path)

    # Create plots
    if save_plots:
        forecast_plot_path = Path(output_dir) / "step7_forecast_distribution.png"
        forecaster.plot_forecast_distribution(save_path=forecast_plot_path)

    # Save forecaster
    if config.SAVE_INTERMEDIATE:
        import pickle
        save_path = Path(config.OUTPUT_DIR) / "step7_forecaster.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(forecaster, f)
        if forecaster.verbose:
            print(f"\nForecaster saved to: {save_path}")

    # Print summary
    print("\n" + "="*60)
    print("STEP 7 COMPLETE - MMAR FORECAST READY!")
    print("="*60)
    print(f"\nVolatility Forecast:")
    print(f"  Point estimate: {results['mean_volatility']:.6f}")
    print(f"  Uncertainty (std dev): {results['std_volatility']:.6f}")
    print(f"  95% Confidence Interval:")
    print(f"    [{results['mean_volatility'] - 1.96*results['std_volatility']:.6f}, "
          f"{results['mean_volatility'] + 1.96*results['std_volatility']:.6f}]")

    print(f"\n" + "="*60)
    print("ALL STEPS COMPLETE!")
    print("="*60)
    print(f"\nYou have successfully built an MMAR volatility forecast model!")
    print(f"\nResults saved in:")
    print(f"  - {results_path}")
    print(f"  - {output_dir}")

    print(f"\nNext steps:")
    print(f"  1. Compare forecast with actual realized volatility")
    print(f"  2. Compare with GARCH forecasts (as in paper)")
    print(f"  3. Use forecast for position sizing or risk management")
    print("="*60 + "\n")

    return forecaster


if __name__ == "__main__":
    print("Step 7: Monte Carlo Volatility Forecasting")
    print("="*60)
    print("\nThis script runs 10,000 MMAR simulations to forecast volatility.")
    print("You must run Steps 1-3 first.\n")
