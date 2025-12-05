"""
Runner script for comparing MMAR forecast with realized volatility.

Usage:
    python run_comparison.py
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

from compare_forecast import run_forecast_comparison
from datetime import datetime, timedelta
import pandas as pd
import config


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "FORECAST VALIDATION")
    print("="*70)

    # Option 1: Use default forecast period (after training data)
    print("\nOption 1: Default forecast period")
    print(f"  Training period ended: {config.END_DATE}")
    training_end = pd.to_datetime(config.END_DATE)
    default_start = (training_end + timedelta(days=1)).strftime("%Y-%m-%d")
    default_end = (training_end + timedelta(days=config.FORECAST_DAYS + 1)).strftime("%Y-%m-%d")
    print(f"  Forecast period: {default_start} to {default_end}")

    # Option 2: Custom period
    print("\nOption 2: Custom forecast period")
    print("  (Edit this script to change dates)")

    # Choose which option
    use_default = True  # Set to False to use custom dates

    if use_default:
        print("\nUsing default forecast period...")
        start_date = None  # Will use day after config.END_DATE
        end_date = None    # Will use start_date + FORECAST_DAYS
        days_ahead = None  # Will use config.FORECAST_DAYS
    else:
        # Custom dates (edit these)
        start_date = "2025-07-02"
        end_date = "2025-07-27"
        days_ahead = 25

    # Run comparison
    try:
        comparison = run_forecast_comparison(
            start_date=start_date,
            end_date=end_date,
            days_ahead=days_ahead,
            save_plots=True
        )

        # Additional analysis
        print("\n" + "="*70)
        print("ADDITIONAL INSIGHTS")
        print("="*70)

        # Annualized volatility
        periods_per_year = 288 * 252  # 5-min periods × trading days
        ann_forecast = comparison.forecast_volatility * (periods_per_year ** 0.5) * 100
        ann_realized = comparison.realized_volatility * (periods_per_year ** 0.5) * 100

        print(f"\nAnnualized Volatility:")
        print(f"  Forecast: {ann_forecast:.2f}%")
        print(f"  Realized: {ann_realized:.2f}%")
        print(f"  Difference: {ann_forecast - ann_realized:.2f}%")

        # Directional accuracy
        if comparison.error > 0:
            print(f"\n  Forecast was HIGHER than realized (overestimated risk)")
        else:
            print(f"\n  Forecast was LOWER than realized (underestimated risk)")

        print("\n" + "="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Make sure Step 7 is complete (python run_step7.py)")
        print(f"  2. Check that forecast period has data available")
        print(f"  3. Verify MT5 connection if using live data")
        print(f"  4. Check config.py settings")
