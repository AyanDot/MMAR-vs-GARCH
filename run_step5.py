"""
Runner script for Step 5: Generate Fractional Brownian Motion

Loads Step 2 results (for H) and generates FBM.
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings


import pickle
from pathlib import Path
import config
from step5_generate_fbm import run_fbm_generation


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*8 + "MMAR STEP 5: GENERATE FRACTIONAL BROWNIAN MOTION")
    print("="*70)

    # Load Step 2 results (need H parameter)
    step2_path = Path(config.OUTPUT_DIR) / "step2_extractor.pkl"

    if not step2_path.exists():
        print(f"\n✗ ERROR: Step 2 results not found!")
        print(f"  Expected file: {step2_path}")
        print(f"\n  You must run Step 2 first:")
        print(f"    python run_step2.py")
        exit(1)

    print(f"\nLoading Step 2 results from: {step2_path}")
    with open(step2_path, 'rb') as f:
        extractor = pickle.load(f)

    print(f"✓ Loaded H = {extractor.H:.4f}")

    # Run Step 5
    # Generate same number of points as cascade (2^10 = 1024)
    fbm_generator = run_fbm_generation(extractor, n_points=2**10, save_plots=True)

    print(f"\n{'='*70}")
    print("Ready for Step 6!")
    print(f"Run: python run_step6.py")
    print(f"{'='*70}\n")
