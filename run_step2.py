"""
STEP 2: Extract Scaling Function τ(q)
Extracts scaling function from Step 1 results and estimates H parameter

PREREQUISITE: Must run run_step1.py first!

USAGE:
1. Make sure you've run run_step1.py successfully
2. Run: python run_step2.py
3. Review results in plots/ folder
4. If successful, proceed to run_step3.py (when implemented)
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings


from step2_extract_scaling import run_scaling_extraction
import config
import pickle
from pathlib import Path
import numpy as np

def main():
    print("\n" + "="*70)
    print("STEP 2: EXTRACT SCALING FUNCTION τ(q)")
    print("="*70)

    # Load checker from Step 1
    print("\n[1/3] Loading Step 1 results...")
    checker_path = Path(config.OUTPUT_DIR) / "step1_checker.pkl"

    if not checker_path.exists():
        print(f"\n✗ Error: Step 1 results not found!")
        print(f"  Looking for: {checker_path}")
        print(f"\nYou must run Step 1 first:")
        print(f"  python run_step1.py")
        return

    try:
        with open(checker_path, 'rb') as f:
            checker = pickle.load(f)
        print(f"  ✓ Loaded checker from: {checker_path}")
    except Exception as e:
        print(f"\n✗ Error loading Step 1 results: {e}")
        print(f"\nTry running Step 1 again:")
        print(f"  python run_step1.py")
        return

    # Verify fractality was confirmed
    if not checker.is_fractal():
        print(f"\n⚠️  WARNING: Step 1 did not confirm fractality")
        # Calculate mean R² from dict values
        mean_r2 = np.mean(list(checker.r_squared_values.values()))
        print(f"  Mean R² = {mean_r2:.4f}")
        print(f"  Threshold = {config.MIN_R_SQUARED}")
        print(f"\n⚠️  Proceeding anyway for exploratory analysis...")
        print(f"  Note: Results may not be meaningful for non-fractal data")

    # Run scaling extraction
    print("\n[2/3] Extracting scaling function...")
    extractor = run_scaling_extraction(checker, save_plots=True)

    # Save extractor for Step 3
    print("\n[3/3] Saving results...")
    extractor_path = Path(config.OUTPUT_DIR) / "step2_extractor.pkl"
    with open(extractor_path, 'wb') as f:
        pickle.dump(extractor, f)
    print(f"  ✓ Extractor object saved to: {extractor_path}")

    # Final summary
    print("\n" + "="*70)
    print("STEP 2 COMPLETE")
    print("="*70)

    print(f"\nKey Results:")
    print(f"  H (self-affinity index) = {extractor.H:.6f}")
    print(f"  τ(q) range = [{extractor.tau_q.min():.4f}, {extractor.tau_q.max():.4f}]")

    print(f"\nInterpretation:")
    if 0.5 < extractor.H < 1.0:
        print(f"  H > 0.5 → Persistent process (long memory)")
    elif abs(extractor.H - 0.5) < 0.05:
        print(f"  H ≈ 0.5 → Random walk")
    elif 0 < extractor.H < 0.5:
        print(f"  H < 0.5 → Anti-persistent process")

    print(f"\nOutput files saved to: {config.PLOT_DIR}/")
    print(f"  - step2_scaling_function.png")
    print(f"  - step2_quality_metrics.png")
    print(f"  - step2_scaling_results.csv")

    print(f"\nNext step:")
    print(f"  Run: python run_step3.py")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
