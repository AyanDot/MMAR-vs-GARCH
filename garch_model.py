"""
GARCH Model Implementation for Comparison with MMAR

Uses the 'arch' library to fit standard GARCH models.
Designed to use the same training data and configuration as MMAR for fair comparison.

Based on paper: Zhang (2017) compared MMAR with:
- Standard GARCH (S-GARCH)
- Exponential GARCH (E-GARCH)
- Integrated GARCH (I-GARCH)
- GJR-GARCH
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from arch import arch_model
import config


class GARCHForecaster:
    """
    GARCH volatility forecasting to compare with MMAR.

    Uses the same training data as MMAR (from Step 1) to ensure
    fair comparison.
    """

    def __init__(self, returns, model_type='GARCH', p=1, q=1, dist='normal', verbose=True):
        """
        Initialize GARCH forecaster.

        Parameters:
        -----------
        returns : np.ndarray
            Training returns (same as used for MMAR)
        model_type : str
            GARCH variant: 'GARCH', 'EGARCH', 'GJR', 'FIGARCH'
        p : int
            GARCH lag order (default 1)
        q : int
            ARCH lag order (default 1)
        dist : str
            Error distribution: 'normal', 't', 'skewt'
        verbose : bool
            Print detailed info
        """
        self.returns = returns * 100  # arch library expects percentage returns
        self.model_type = model_type
        self.p = p
        self.q = q
        self.dist = dist
        self.verbose = verbose

        # Storage
        self.model = None
        self.fit_result = None
        self.forecast_variance = None
        self.forecast_volatility = None

        if self.verbose:
            print(f"\nGARCH Forecaster Initialized")
            print(f"Model: {model_type}({p},{q})")
            print(f"Distribution: {dist}")
            print(f"Training returns: {len(returns)}")

    def fit(self):
        """
        Fit GARCH model to training data.
        """
        if self.verbose:
            print(f"\nFitting {self.model_type}({self.p},{self.q}) model...")

        try:
            # Create model
            # CRITICAL: Use rescale=True to handle small return values
            # The arch library will automatically rescale for numerical stability
            # then convert forecasts back to original scale
            self.model = arch_model(
                self.returns,
                vol=self.model_type,
                p=self.p,
                q=self.q,
                dist=self.dist,
                rescale=True  # Automatically rescale for numerical stability
            )

            # Fit model with options for better convergence
            self.fit_result = self.model.fit(
                disp='off',
                options={'ftol': 1e-6, 'maxiter': 1000}
            )

            if self.verbose:
                print(f"  ✓ Model fitted successfully")
                print(f"\nModel Summary:")
                print(f"  Log-Likelihood: {self.fit_result.loglikelihood:.2f}")
                print(f"  AIC: {self.fit_result.aic:.2f}")
                print(f"  BIC: {self.fit_result.bic:.2f}")

            return self.fit_result

        except Exception as e:
            print(f"  ✗ Error fitting {self.model_type}: {e}")
            raise

    def forecast(self, n_sim=10000):
        """
        Generate volatility forecast using Zhang's EXACT methodology from paper.

        From paper (Section 5.c):
        "The 'ugarchfit' and 'ugarchsim' functions were used to fit a GARCH
        model to the data, and simulate 10,000 returns. The standard deviation
        of these returns is taken as a forecast of volatility."

        CRITICAL DIFFERENCE vs MMAR:
        - MMAR: 10,000 simulations × 1024 returns each → mean of 10,000 std devs
        - GARCH: ONE simulation × 10,000 returns → std dev once

        This is Zhang's method, NOT the same as MMAR's Monte Carlo approach!

        Parameters:
        -----------
        n_sim : int
            Number of returns to simulate (default 10000, as per paper)

        Returns:
        --------
        float
            Forecasted volatility (std dev of simulated returns)
        """
        if self.fit_result is None:
            raise ValueError("Model must be fitted before forecasting. Call fit() first.")

        if self.verbose:
            print(f"\nGenerating GARCH forecast (Zhang's method)...")
            print(f"  Simulating {n_sim} returns from fitted model")

        # Simulate ONE path of n_sim returns
        # This matches ugarchsim(fit, n.sim=10000) from the paper
        sim = self.fit_result.forecast(horizon=n_sim, method='simulation', simulations=1)

        # Extract the simulated returns
        # Shape: (1 simulation, 1 starting point, n_sim returns)
        sim_returns = sim.simulations.values[0, 0, :]  # Shape: (n_sim,)

        # CRITICAL FIX: If model was rescaled, simulated returns are in rescaled space
        # We need to unscale them before calculating std dev
        if hasattr(self.model, 'scale') and self.model.scale is not None:
            # Unscale the simulated returns
            sim_returns = sim_returns / self.model.scale
            if self.verbose:
                print(f"  Unscaling by factor: {self.model.scale:.6f}")

        # Calculate std dev of these returns (still in percentage form)
        forecast_vol_pct = np.std(sim_returns)

        # Convert back to original units (from percentage)
        self.forecast_volatility = forecast_vol_pct / 100

        if self.verbose:
            print(f"  ✓ Simulation complete")
            print(f"  Simulated {len(sim_returns)} returns")
            print(f"  Forecast volatility: {self.forecast_volatility:.10f}")
            print(f"\n  Method note: This is ONE simulation of {n_sim} returns")
            print(f"               (NOT {n_sim} simulations like MMAR)")

        return self.forecast_volatility

    def get_summary(self):
        """Get model fit summary."""
        if self.fit_result is None:
            return "Model not fitted yet"
        return self.fit_result.summary()


def run_garch_comparison(model_types=['GARCH', 'EGARCH', 'GJR'],
                         p=1, q=1, dist='normal',
                         output_dir=None, verbose=True):
    """
    Run multiple GARCH models for comparison with MMAR.

    Uses the SAME training data as MMAR (from Step 1) for fair comparison.

    Parameters:
    -----------
    model_types : list
        List of GARCH variants to fit: 'GARCH', 'EGARCH', 'GJR', 'FIGARCH'
    p : int
        GARCH lag order
    q : int
        ARCH lag order
    dist : str
        Error distribution: 'normal', 't', 'skewt'
    output_dir : str, optional
        Directory to save results
    verbose : bool
        Print detailed info

    Returns:
    --------
    dict
        Dictionary of fitted GARCH models
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load training data from Step 1 (same as MMAR uses)
    step1_path = Path(config.OUTPUT_DIR) / "step1_checker.pkl"

    if not step1_path.exists():
        raise FileNotFoundError(
            f"Step 1 results not found: {step1_path}\n"
            f"You must run Step 1 first to get training data."
        )

    print("\n" + "="*70)
    print(" "*20 + "GARCH MODEL COMPARISON")
    print("="*70)

    with open(step1_path, 'rb') as f:
        checker = pickle.load(f)

    returns = checker.returns

    print(f"\nUsing SAME training data as MMAR:")
    print(f"  Period: {config.START_DATE} to {config.END_DATE}")
    print(f"  Number of returns: {len(returns)}")
    print(f"  Mean: {np.mean(returns):.10f}")
    print(f"  Std dev: {np.std(returns):.10f}")

    # Fit each GARCH model
    results = {}

    for model_type in model_types:
        print(f"\n" + "="*70)
        print(f"FITTING {model_type}({p},{q}) MODEL")
        print("="*70)

        try:
            forecaster = GARCHForecaster(
                returns=returns,
                model_type=model_type,
                p=p,
                q=q,
                dist=dist,
                verbose=verbose
            )

            # Fit model
            forecaster.fit()

            # Generate forecast using Zhang's EXACT methodology from paper
            # Simulate ONE path of 10,000 returns, take std dev
            forecast_vol = forecaster.forecast(n_sim=10000)

            # Store results
            results[model_type] = {
                'forecaster': forecaster,
                'forecast_volatility': forecast_vol,
                'aic': forecaster.fit_result.aic,
                'bic': forecaster.fit_result.bic,
                'loglik': forecaster.fit_result.loglikelihood
            }

        except Exception as e:
            print(f"  ✗ Failed to fit {model_type}: {e}")
            results[model_type] = None

    # Save results
    results_path = Path(output_dir) / "garch_models.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    if verbose:
        print(f"\n✓ GARCH models saved to: {results_path}")

    # Print comparison summary
    print("\n" + "="*70)
    print("GARCH MODELS SUMMARY")
    print("="*70)

    print(f"\n{'Model':<15} {'Forecast Vol':<15} {'AIC':<12} {'BIC':<12} {'LogLik':<12}")
    print("-"*70)

    for model_type, result in results.items():
        if result is not None:
            print(f"{model_type:<15} "
                  f"{result['forecast_volatility']:.10f}  "
                  f"{result['aic']:>10.2f}  "
                  f"{result['bic']:>10.2f}  "
                  f"{result['loglik']:>10.2f}")
        else:
            print(f"{model_type:<15} FAILED")

    print("="*70)

    return results


if __name__ == "__main__":
    print("GARCH Model Implementation")
    print("="*70)
    print("\nThis module fits GARCH models using the same training data as MMAR")
    print("for fair comparison.")
    print("\nUsage:")
    print("  from garch_model import run_garch_comparison")
    print("  results = run_garch_comparison()")
    print("\nOr run the comparison script:")
    print("  python run_garch_comparison.py")
