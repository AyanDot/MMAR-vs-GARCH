"""
Runner script for Step 7: Monte Carlo Volatility Forecasting

Loads Step 3 results and runs 10,000 MMAR simulations.
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings


import pickle
from pathlib import Path
import numpy as np
import config
from step7_monte_carlo import run_monte_carlo_forecast


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*8 + "MMAR STEP 7: MONTE CARLO VOLATILITY FORECASTING")
    print("="*70)

    # Load Step 3 results (need fitter for distribution params)
    step3_path = Path(config.OUTPUT_DIR) / "step3_fitter.pkl"

    if not step3_path.exists():
        print(f"\n✗ ERROR: Step 3 results not found!")
        print(f"  Expected file: {step3_path}")
        print(f"\n  You must run Step 3 first:")
        print(f"    python run_step3.py")
        exit(1)

    print(f"\nLoading Step 3 results from: {step3_path}")
    with open(step3_path, 'rb') as f:
        fitter = pickle.load(f)

    print(f"✓ Loaded distribution: {fitter.best_distribution}")
    print(f"✓ H = {fitter.H:.4f}")

    # Load historical returns from Step 1 to get actual sample volatility
    step1_path = Path(config.OUTPUT_DIR) / "step1_checker.pkl"

    if not step1_path.exists():
        print(f"\n✗ ERROR: Step 1 results not found!")
        print(f"  Expected file: {step1_path}")
        print(f"\n  You must run Step 1 first:")
        print(f"    python run_step1.py")
        exit(1)

    with open(step1_path, 'rb') as f:
        checker = pickle.load(f)

    # Calculate sample volatility from actual training data
    sample_volatility = np.std(checker.returns)

    print(f"\nUsing sample volatility from training data:")
    print(f"  Training period: {config.START_DATE} to {config.END_DATE}")
    print(f"  Sample volatility: {sample_volatility:.10f}")
    print(f"  (Std dev of {len(checker.returns)} historical returns)")

    # Ask user for number of simulations
    print(f"\nDefault: {config.NUM_SIMULATIONS} simulations")
    print(f"⚠️  WARNING: 10,000 simulations will take 10-30 minutes!")
    print(f"\nFor testing, you can run fewer simulations (e.g., 100)")

    # Run Step 7
    forecaster = run_monte_carlo_forecast(fitter,
                                         sample_volatility=sample_volatility,
                                         n_simulations=config.NUM_SIMULATIONS,
                                         save_plots=True)

    print(f"\n{'='*70}")
    print("🎉 CONGRATULATIONS! 🎉")
    print("="*70)
    print("\nYou have completed all 7 steps of the MMAR model!")
    print("\nYour volatility forecast:")
    print(f"  Point estimate: {forecaster.mean_forecast:.6f}")
    print(f"  Uncertainty: ±{1.96 * forecaster.std_forecast:.6f} (95% CI)")
    print(f"\nNext steps:")
    print(f"  1. Compare with realized volatility")
    print(f"  2. Compare with GARCH models")
    print(f"  3. Use for position sizing / risk management")
    print("="*70 + "\n")
