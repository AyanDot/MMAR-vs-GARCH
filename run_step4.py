"""
Runner script for Step 4: Generate Multifractal Cascade

Loads Step 3 results and generates trading time.
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings


import pickle
from pathlib import Path
import config
from step4_generate_cascade import run_cascade_generation


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*10 + "MMAR STEP 4: GENERATE MULTIFRACTAL CASCADE")
    print("="*70)

    # Load Step 3 results
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

    # Run Step 4
    generator = run_cascade_generation(fitter, save_plots=True, b=2, k=10)

    print(f"\n{'='*70}")
    print("Ready for Step 5!")
    print(f"Run: python run_step5.py")
    print(f"{'='*70}\n")
