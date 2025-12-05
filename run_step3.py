"""
Runner script for Step 3: Fit Multifractal Spectrum

Loads Step 2 results and fits spectrum to 4 distributions.
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings


import pickle
from pathlib import Path
import config
from step3_fit_spectrum import run_spectrum_fitting


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "MMAR STEP 3: FIT MULTIFRACTAL SPECTRUM")
    print("="*70)

    # Load Step 2 results
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

    # Run Step 3
    fitter = run_spectrum_fitting(extractor, save_plots=True)

    # Save for Step 4
    save_path = Path(config.OUTPUT_DIR) / "step3_fitter.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(fitter, f)

    print(f"\n✓ Step 3 results saved to: {save_path}")
    print(f"\n{'='*70}")
    print("Ready for Step 4!")
    print(f"Run: python run_step4.py")
    print(f"{'='*70}\n")
