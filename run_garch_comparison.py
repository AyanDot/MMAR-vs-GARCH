"""
Runner script for GARCH vs MMAR comparison.

Fits multiple GARCH models and compares forecasts with MMAR and realized volatility.
"""

from garch_model import run_garch_comparison
from compare_forecast import run_forecast_comparison
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import config


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "MMAR vs GARCH COMPARISON")
    print("="*70)

    # Step 1: Fit GARCH models
    print("\n[STEP 1] Fitting GARCH models...")
    print("-"*70)

    garch_results = run_garch_comparison(
        model_types=['GARCH', 'EGARCH', 'GJR'],
        p=1,
        q=1,
        dist='normal',
        verbose=True
    )

    # Step 2: Load MMAR forecast
    print("\n" + "="*70)
    print("[STEP 2] Loading MMAR forecast...")
    print("-"*70)

    mmar_path = Path(config.OUTPUT_DIR) / "step7_forecaster.pkl"
    if not mmar_path.exists():
        print(f"✗ MMAR forecast not found: {mmar_path}")
        print(f"  You must run Step 7 first: python run_step7.py")
        exit(1)

    with open(mmar_path, 'rb') as f:
        mmar_forecaster = pickle.load(f)

    mmar_forecast = mmar_forecaster.mean_forecast

    print(f"✓ MMAR forecast loaded: {mmar_forecast:.10f}")

    # Step 3: Compare with realized volatility
    print("\n" + "="*70)
    print("[STEP 3] Comparing with realized volatility...")
    print("-"*70)

    # Run forecast comparison to get realized volatility
    try:
        comparison = run_forecast_comparison(
            start_date=None,
            end_date=None,
            days_ahead=None,
            save_plots=False
        )

        realized_vol = comparison.realized_volatility

        print(f"✓ Realized volatility: {realized_vol:.10f}")

        # Step 4: Comparison table
        print("\n" + "="*70)
        print("FULL COMPARISON: MMAR vs GARCH vs REALIZED")
        print("="*70)

        # Prepare comparison data
        comparison_data = []

        # Add MMAR
        mmar_error = abs(mmar_forecast - realized_vol)
        mmar_pct_error = 100 * mmar_error / realized_vol
        comparison_data.append({
            'Model': 'MMAR',
            'Forecast': mmar_forecast,
            'Error': mmar_error,
            'Error %': mmar_pct_error,
            'AIC': np.nan,
            'BIC': np.nan
        })

        # Add GARCH models
        for model_type, result in garch_results.items():
            if result is not None:
                forecast = result['forecast_volatility']
                error = abs(forecast - realized_vol)
                pct_error = 100 * error / realized_vol

                comparison_data.append({
                    'Model': model_type,
                    'Forecast': forecast,
                    'Error': error,
                    'Error %': pct_error,
                    'AIC': result['aic'],
                    'BIC': result['bic']
                })

        # Create DataFrame and sort by error
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Error %')

        print(f"\n{'Model':<15} {'Forecast':<15} {'Error':<15} {'Error %':<10} {'AIC':<12} {'BIC':<12}")
        print("-"*85)

        for _, row in df.iterrows():
            aic_str = f"{row['AIC']:>10.2f}" if not np.isnan(row['AIC']) else "     N/A"
            bic_str = f"{row['BIC']:>10.2f}" if not np.isnan(row['BIC']) else "     N/A"

            print(f"{row['Model']:<15} "
                  f"{row['Forecast']:.10f}  "
                  f"{row['Error']:.10f}  "
                  f"{row['Error %']:>7.2f}%  "
                  f"{aic_str}  "
                  f"{bic_str}")

        print("-"*85)
        print(f"{'REALIZED':<15} {realized_vol:.10f}")
        print("="*85)

        # Determine winner
        best_model = df.iloc[0]['Model']
        best_error = df.iloc[0]['Error %']

        print(f"\n🏆 BEST MODEL: {best_model} (Error: {best_error:.2f}%)")

        # Statistical comparison (from paper)
        print(f"\n" + "="*70)
        print("STATISTICAL COMPARISON")
        print("="*70)

        mmar_row = df[df['Model'] == 'MMAR'].iloc[0]
        print(f"\nMMar Performance:")
        print(f"  Forecast: {mmar_row['Forecast']:.10f}")
        print(f"  Error: {mmar_row['Error %']:.2f}%")

        # Compare MMAR vs each GARCH
        for model_type in ['GARCH', 'EGARCH', 'GJR']:
            if model_type in df['Model'].values:
                garch_row = df[df['Model'] == model_type].iloc[0]
                diff = mmar_row['Error %'] - garch_row['Error %']

                if diff < 0:
                    print(f"\n  MMAR vs {model_type}: MMAR is {abs(diff):.2f}% better ✓")
                else:
                    print(f"\n  MMAR vs {model_type}: {model_type} is {diff:.2f}% better")

        print("\n" + "="*70)
        print("PAPER RESULTS (Zhang 2017):")
        print("  MMAR significantly outperformed GARCH models")
        print("  This was shown via lower RMSE across 20 stocks")
        print("="*70)

        # Save comparison results
        results_path = Path(config.PLOT_DIR) / "mmar_vs_garch_comparison.csv"
        df['Realized'] = realized_vol
        df.to_csv(results_path, index=False)
        print(f"\n✓ Comparison saved to: {results_path}")

    except Exception as e:
        print(f"\n✗ Error during comparison: {e}")
        print(f"\nMake sure forecast period data is available in MT5")

    print("\n" + "="*70 + "\n")
