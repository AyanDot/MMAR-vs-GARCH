"""
Runner script for Step 6: Combine FBM with Trading Time

Loads Step 4 (cascade) and Step 5 (FBM) results and combines them.
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings


import pickle
from pathlib import Path
import config
from step6_combine_model import run_mmar_combination


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*12 + "MMAR STEP 6: COMBINE FBM WITH TRADING TIME")
    print("="*70)

    # Load Step 4 results (cascade)
    step4_path = Path(config.OUTPUT_DIR) / "step4_generator.pkl"

    if not step4_path.exists():
        print(f"\n✗ ERROR: Step 4 results not found!")
        print(f"  Expected file: {step4_path}")
        print(f"\n  You must run Step 4 first:")
        print(f"    python run_step4.py")
        exit(1)

    print(f"\nLoading Step 4 results from: {step4_path}")
    with open(step4_path, 'rb') as f:
        cascade_generator = pickle.load(f)

    print(f"✓ Loaded cascade with {len(cascade_generator.trading_time)} points")

    # Load Step 5 results (FBM)
    step5_path = Path(config.OUTPUT_DIR) / "step5_fbm_generator.pkl"

    if not step5_path.exists():
        print(f"\n✗ ERROR: Step 5 results not found!")
        print(f"  Expected file: {step5_path}")
        print(f"\n  You must run Step 5 first:")
        print(f"    python run_step5.py")
        exit(1)

    print(f"Loading Step 5 results from: {step5_path}")
    with open(step5_path, 'rb') as f:
        fbm_generator = pickle.load(f)

    print(f"✓ Loaded FBM with H = {fbm_generator.H:.4f}")

    # Run Step 6
    combiner = run_mmar_combination(fbm_generator, cascade_generator, save_plots=True)

    print(f"\n{'='*70}")
    print("Ready for Step 7!")
    print(f"Run: python run_step7.py")
    print(f"{'='*70}\n")
