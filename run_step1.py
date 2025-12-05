"""
STEP 1: Check for Fractality
Runs partition function analysis to detect multifractal behavior

USAGE:
1. Make sure MT5 terminal is running
2. Run: python run_step1.py
3. Review results in plots/ folder
4. If fractality confirmed, proceed to run_step2.py
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings


from data_loader import DataLoader
from step1_check_fractality import run_fractality_check
import config
import pickle
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("STEP 1: CHECK FOR FRACTALITY")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Symbol: {config.SYMBOL}")
    print(f"  Timeframe: {config.TIMEFRAME_MT5}")
    print(f"  Period: {config.START_DATE} to {config.END_DATE}")

    # Load data
    print("\n[1/4] Loading data from MetaTrader 5...")
    loader = DataLoader(
        symbol=config.SYMBOL,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        verbose=True
    )

    try:
        loader.load_from_mt5()
    except Exception as e:
        print(f"\n✗ Failed to load from MT5: {e}")
        print("\nTroubleshooting:")
        print("  1. Is MetaTrader 5 terminal running?")
        print("  2. Are you logged into an account?")
        print("  3. Is the symbol available?")
        return

    # Calculate returns
    print("\n[2/4] Calculating log returns...")
    loader.calculate_returns(price_column='close', method='log')
    loader.summary_statistics()

    # Get returns
    returns = loader.get_returns_array()
    print(f"\n✓ Ready: {len(returns):,} returns loaded")

    # Run fractality check
    print("\n[3/4] Running fractality analysis...")
    checker = run_fractality_check(returns, save_plots=True)

    # Save checker object for Step 2
    print("\n[4/4] Saving results...")
    results_dir = Path(config.OUTPUT_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    checker_path = results_dir / "step1_checker.pkl"
    with open(checker_path, 'wb') as f:
        pickle.dump(checker, f)
    print(f"  ✓ Checker object saved to: {checker_path}")

    # Final summary
    print("\n" + "="*70)
    print("STEP 1 COMPLETE")
    print("="*70)

    results_df = checker.get_results_dataframe()
    mean_r2 = results_df['r_squared'].mean()

    print(f"\nResults:")
    print(f"  Mean R² = {mean_r2:.4f}")
    print(f"  Threshold = {config.MIN_R_SQUARED}")

    if checker.is_fractal():
        print(f"\n✓ FRACTALITY CONFIRMED")
        print(f"\nNext step:")
        print(f"  Run: python run_step2.py")
    else:
        print(f"\n✗ WEAK FRACTALITY")
        print(f"\nData may not be suitable for MMAR.")
        print(f"Consider adjusting parameters or trying different data.")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
