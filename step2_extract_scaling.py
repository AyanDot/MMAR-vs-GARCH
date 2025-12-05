"""
Step 2: Extract Scaling Function τ(q)
Extracts the scaling function from Step 1 partition function slopes

From paper equation (2.1):
E(|X(t)|^q) = c(q) × t^(τ(q)+1)

Therefore:
slope = τ(q) + 1  →  τ(q) = slope - 1

Key parameter to estimate:
H (self-affinity index) where τ(1/H) = 0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from pathlib import Path
import config

# Try to import nolds for robust Hurst estimation
try:
    import nolds
    NOLDS_AVAILABLE = True
except ImportError:
    NOLDS_AVAILABLE = False
    print("⚠️  Warning: nolds not available. Install with: pip install nolds")


class ScalingExtractor:
    """
    Extracts scaling function τ(q) from partition function results.

    The scaling function τ(q) characterizes the multifractal properties
    of the data. It is extracted from the slopes of partition plots:

    τ(q) = slope - 1

    Key parameter:
    H (Hurst exponent / self-affinity index) is found where τ(1/H) = 0
    """

    def __init__(self, checker, verbose=config.VERBOSE):
        """
        Initialize scaling extractor.

        Parameters:
        -----------
        checker : FractalityChecker
            Completed checker object from Step 1 with slopes computed
        verbose : bool
            Print detailed information
        """
        self.checker = checker
        self.verbose = verbose

        # Extract from checker
        self.q_values = np.array(sorted(checker.slopes.keys()))
        self.slopes = np.array([checker.slopes[q] for q in self.q_values])
        self.r_squared = np.array([checker.r_squared_values[q] for q in self.q_values])

        # Calculate τ(q) = slope - 1
        self.tau_q = self.slopes - 1

        # Storage for results
        self.H = None  # Self-affinity index
        self.tau_q_func = None  # Interpolation function for τ(q)

        if self.verbose:
            print(f"\nScaling Extractor Initialized")
            print(f"Number of q values: {len(self.q_values)}")
            print(f"q range: {self.q_values.min():.2f} to {self.q_values.max():.2f}")
            print(f"τ(q) range: {self.tau_q.min():.4f} to {self.tau_q.max():.4f}")

    def estimate_H_nolds(self):
        """
        Estimate H using nolds R/S (rescaled range) method.

        This is more robust than finding τ(1/H) = 0 numerically.
        Uses the original returns from the checker.

        Returns:
        --------
        float
            Estimated H value using R/S method
        """
        if not NOLDS_AVAILABLE:
            if self.verbose:
                print("  ⚠️  nolds not available, falling back to τ(q) method")
            return None

        if self.verbose:
            print("\nEstimating H using nolds R/S method...")
            print("  (More robust than τ(1/H) = 0 approach)")

        try:
            # Get returns from checker
            returns = self.checker.returns

            # Estimate Hurst using nolds
            # corrected=True applies Anis-Lloyd-Peters correction
            # fit='RANSAC' is robust to outliers
            H_nolds = nolds.hurst_rs(returns, fit='RANSAC', corrected=True)

            if self.verbose:
                print(f"  ✓ H estimated: {H_nolds:.4f}")

            return H_nolds

        except Exception as e:
            if self.verbose:
                print(f"  ✗ nolds estimation failed: {e}")
            return None

    def _print_H_interpretation(self):
        """Print interpretation of estimated H value."""
        if not self.verbose or self.H is None:
            return

        print(f"\n  Interpretation of H = {self.H:.4f}:")

        # Check if close to 0.5
        if abs(self.H - 0.5) < 0.05:
            print(f"  → H ≈ 0.5: Close to random walk")
            print(f"    (No strong long-term memory)")
        elif self.H > 0.5:
            if self.H > 0.7:
                print(f"  → H > 0.7: STRONGLY persistent")
                print(f"    (Strong trends, long memory)")
            else:
                print(f"  → H > 0.5: Persistent process")
                print(f"    (Moderate trends, some long memory)")
        elif self.H < 0.5:
            if self.H < 0.3:
                print(f"  → H < 0.3: STRONGLY anti-persistent")
                print(f"    (Sharp mean reversion)")
            else:
                print(f"  → H < 0.5: Anti-persistent")
                print(f"    (Mean-reverting behavior)")

        # Additional context
        print(f"\n  Context:")
        print(f"  - Random walk: H = 0.50")
        print(f"  - FX markets typically: H = 0.48-0.55")
        print(f"  - Stock markets typically: H = 0.52-0.58")

        # Warning if extreme
        if self.H > 0.8 or self.H < 0.2:
            print(f"\n  ⚠️  EXTREME H value!")
            print(f"     This is unusual for financial data")
            print(f"     → Check data quality")
            print(f"     → May indicate measurement error")

    def estimate_H(self, use_nolds=True):
        """
        Estimate H (self-affinity index).

        If use_nolds=True, uses robust R/S method from nolds library.
        Otherwise, finds where τ(1/H) = 0 from scaling function.

        Parameters:
        -----------
        use_nolds : bool
            If True, use nolds.hurst_rs() method (recommended)

        Returns:
        --------
        float
            Estimated H value
        """
        # Try nolds first (more robust)
        if use_nolds and NOLDS_AVAILABLE:
            H_nolds = self.estimate_H_nolds()
            if H_nolds is not None:
                self.H = H_nolds
                self._print_H_interpretation()
                return self.H

        # Fallback to τ(1/H) = 0 method
        if self.verbose:
            print("\nEstimating H from τ(1/H) = 0...")
            print("  (Finding q where τ(q) = 0)")

        # Create interpolation function for τ(q)
        # Use cubic spline for smooth interpolation
        self.tau_q_func = interp1d(self.q_values, self.tau_q,
                                    kind='cubic',
                                    fill_value='extrapolate',
                                    bounds_error=False)

        # Find where τ(q) = 0
        # τ(q) should cross zero somewhere between q_min and q_max
        try:
            # Use Brent's method to find root
            # τ(q) should be positive for small q and negative for large q
            # or vice versa depending on data

            # Check if we have a sign change
            tau_min = self.tau_q_func(self.q_values.min())
            tau_max = self.tau_q_func(self.q_values.max())

            if tau_min * tau_max > 0:
                # No sign change - τ(q) doesn't cross zero in our range
                # This can happen with limited q range
                if self.verbose:
                    print(f"  ⚠️  No zero crossing found in q range")
                    print(f"  τ(q_min={self.q_values.min():.2f}) = {tau_min:.4f}")
                    print(f"  τ(q_max={self.q_values.max():.2f}) = {tau_max:.4f}")

                # Estimate H using linear extrapolation
                # Find q where τ(q) would be closest to 0
                idx_closest = np.argmin(np.abs(self.tau_q))
                q_zero = self.q_values[idx_closest]
                self.H = 1.0 / q_zero

                if self.verbose:
                    print(f"  Using closest point: q = {q_zero:.4f}, τ(q) = {self.tau_q[idx_closest]:.4f}")

            else:
                # Find zero crossing using Brent's method
                q_zero = brentq(self.tau_q_func, self.q_values.min(), self.q_values.max())
                self.H = 1.0 / q_zero

            if self.verbose:
                print(f"  ✓ Found: q = {q_zero:.4f} where τ(q) ≈ 0")
                print(f"  ✓ H = 1/q = {self.H:.4f}")

            # Print interpretation
            self._print_H_interpretation()

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Error estimating H: {e}")
                print(f"  Using fallback estimation method")

            # Fallback: linear fit near τ(q) = 0
            # Find the two points closest to τ = 0
            idx = np.argsort(np.abs(self.tau_q))[:2]
            q1, q2 = self.q_values[idx]
            tau1, tau2 = self.tau_q[idx]

            # Linear interpolation
            if tau2 != tau1:
                q_zero = q1 + (0 - tau1) * (q2 - q1) / (tau2 - tau1)
                self.H = 1.0 / q_zero
            else:
                # Use average
                q_zero = (q1 + q2) / 2
                self.H = 1.0 / q_zero

            if self.verbose:
                print(f"  Estimated H = {self.H:.4f} (fallback method)")

        return self.H

    def validate_scaling_function(self):
        """
        Validate that τ(q) has expected properties.

        From paper:
        1. τ(q) should be concave (d²τ/dq² < 0)
        2. τ(0) = 0 (by definition)
        3. τ(1) relates to mean behavior
        4. τ(q) should be smooth
        """
        if self.verbose:
            print("\nValidating scaling function properties...")

        issues = []

        # Check concavity (second derivative should be negative)
        d2_tau = np.diff(self.tau_q, 2)  # Second differences approximate second derivative
        concave_pct = 100 * np.sum(d2_tau < 0) / len(d2_tau)

        if concave_pct < 70:
            issues.append(f"Low concavity: only {concave_pct:.1f}% of points are concave")

        # Check τ(q≈0) is reasonable
        # Theoretically: τ(0) = -1 (by normalization in some formulations)
        # In practice: τ(q≈0) should be close to -1.0 (between -0.8 and -1.2 is OK)
        if self.q_values.min() <= 0.1:
            idx_near_zero = np.argmin(np.abs(self.q_values))
            q_near_zero = self.q_values[idx_near_zero]
            tau_at_zero = self.tau_q[idx_near_zero]
            # Check if it's far from expected -1.0
            if tau_at_zero < -1.5 or tau_at_zero > -0.5:
                issues.append(f"τ(q≈{q_near_zero:.2f}) = {tau_at_zero:.4f} (expected ≈ -1.0, acceptable range: -0.5 to -1.5)")

        # Check smoothness (no large jumps)
        d_tau = np.diff(self.tau_q)
        max_jump = np.max(np.abs(d_tau))
        if max_jump > 1.0:
            issues.append(f"Large jump in τ(q): {max_jump:.4f}")

        if issues:
            print("  ⚠️  Potential issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  ✓ Scaling function looks good")
            print(f"    Concavity: {concave_pct:.1f}% of points")

        return len(issues) == 0

    def check_multifractality(self):
        """
        CRITICAL TEST: Determine if data is truly MULTIFRACTAL vs monofractal.

        This is THE decisive test for MMAR viability!

        Key principle:
        - Linear τ(q) → Monofractal (e.g., random walk) → MMAR will fail
        - Concave τ(q) → Multifractal → MMAR worth pursuing

        Returns:
        --------
        dict
            Dictionary with multifractality assessment and scores
        """
        print("\n" + "="*60)
        print("CRITICAL TEST: MULTIFRACTALITY CHECK")
        print("="*60)

        # Test 1: Concavity (second derivative)
        d2_tau = np.diff(self.tau_q, 2)
        concave_pct = 100 * np.sum(d2_tau < 0) / len(d2_tau)
        mean_d2 = np.mean(d2_tau)

        print(f"\n1. Concavity Test:")
        print(f"   τ(q) concave at {concave_pct:.1f}% of points")
        print(f"   Mean d²τ/dq² = {mean_d2:.6f}")

        if concave_pct >= 80:
            concavity_verdict = "STRONGLY CONCAVE ✓✓"
            concavity_score = 3
        elif concave_pct >= 60:
            concavity_verdict = "MODERATELY CONCAVE ✓"
            concavity_score = 2
        elif concave_pct >= 40:
            concavity_verdict = "WEAKLY CONCAVE ~"
            concavity_score = 1
        else:
            concavity_verdict = "NOT CONCAVE ✗"
            concavity_score = 0

        print(f"   → {concavity_verdict}")

        # Test 2: Compare with theoretical random walk
        # For Brownian motion: τ(q) = q*H - 1 where H = 0.5
        # So τ(q) = 0.5*q - 1 (perfectly linear)
        q_range = self.q_values[self.q_values > 0]
        tau_range = self.tau_q[self.q_values > 0]

        # Fit linear model
        from scipy.stats import linregress
        slope_linear, intercept_linear, r_value_linear, _, _ = linregress(q_range, tau_range)
        r2_linear = r_value_linear ** 2

        # Calculate residuals from linear fit
        tau_linear_fit = slope_linear * q_range + intercept_linear
        rmse_from_linear = np.sqrt(np.mean((tau_range - tau_linear_fit)**2))

        print(f"\n2. Linearity Test (random walk comparison):")
        print(f"   R² of linear fit = {r2_linear:.4f}")
        print(f"   RMSE from linear = {rmse_from_linear:.6f}")

        if r2_linear > 0.99 and rmse_from_linear < 0.05:
            linearity_verdict = "HIGHLY LINEAR (like random walk) ✗"
            linearity_score = 0
        elif r2_linear > 0.95:
            linearity_verdict = "MOSTLY LINEAR (weakly multifractal) ~"
            linearity_score = 1
        elif r2_linear > 0.90:
            linearity_verdict = "MODERATELY NONLINEAR ✓"
            linearity_score = 2
        else:
            linearity_verdict = "STRONGLY NONLINEAR ✓✓"
            linearity_score = 3

        print(f"   → {linearity_verdict}")

        # Test 3: Range of τ(q) derivatives
        # Multifractals have varying local slopes
        d_tau = np.diff(self.tau_q)
        slope_range = np.ptp(d_tau)  # Peak-to-peak
        slope_std = np.std(d_tau)

        print(f"\n3. Slope Variation Test:")
        print(f"   Range of dτ/dq = {slope_range:.4f}")
        print(f"   Std dev of dτ/dq = {slope_std:.4f}")

        if slope_range > 0.5:
            variation_verdict = "HIGH VARIATION ✓✓"
            variation_score = 3
        elif slope_range > 0.3:
            variation_verdict = "MODERATE VARIATION ✓"
            variation_score = 2
        elif slope_range > 0.1:
            variation_verdict = "LOW VARIATION ~"
            variation_score = 1
        else:
            variation_verdict = "MINIMAL VARIATION (monofractal) ✗"
            variation_score = 0

        print(f"   → {variation_verdict}")

        # Overall verdict
        total_score = concavity_score + linearity_score + variation_score
        max_score = 9

        print(f"\n" + "="*60)
        print(f"OVERALL MULTIFRACTALITY SCORE: {total_score}/{max_score}")
        print("="*60)

        if total_score >= 7:
            overall = "STRONGLY MULTIFRACTAL ✓✓"
            mmar_recommendation = "MMAR highly recommended - proceed with confidence"
        elif total_score >= 5:
            overall = "MODERATELY MULTIFRACTAL ✓"
            mmar_recommendation = "MMAR worth trying - may outperform GARCH"
        elif total_score >= 3:
            overall = "WEAKLY MULTIFRACTAL ~"
            mmar_recommendation = "MMAR uncertain - run comparison but GARCH may win"
        else:
            overall = "NOT MULTIFRACTAL (Monofractal) ✗"
            mmar_recommendation = "MMAR NOT recommended - data is too close to random walk"

        print(f"\n{overall}")
        print(f"→ {mmar_recommendation}")
        print("="*60 + "\n")

        return {
            'concavity_pct': concave_pct,
            'concavity_score': concavity_score,
            'linear_r2': r2_linear,
            'linearity_score': linearity_score,
            'slope_range': slope_range,
            'variation_score': variation_score,
            'total_score': total_score,
            'max_score': max_score,
            'is_multifractal': total_score >= 5,
            'recommendation': mmar_recommendation
        }

    def get_tau_q_at_q(self, q):
        """
        Get τ(q) value at specific q (uses interpolation).

        Parameters:
        -----------
        q : float or array-like
            Moment order(s)

        Returns:
        --------
        float or np.ndarray
            τ(q) value(s)
        """
        if self.tau_q_func is None:
            self.tau_q_func = interp1d(self.q_values, self.tau_q,
                                       kind='cubic',
                                       fill_value='extrapolate',
                                       bounds_error=False)

        return self.tau_q_func(q)

    def plot_scaling_function(self, save_path=None):
        """
        Plot τ(q) vs q.

        Shows the scaling function extracted from partition plots.
        Key features:
        - Should be concave
        - Crosses zero at q = 1/H
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: τ(q) with data points and fit
        ax1.plot(self.q_values, self.tau_q, 'o-',
                linewidth=2, markersize=6,
                label='τ(q) = slope - 1', color='steelblue')

        # Add horizontal line at τ = 0
        ax1.axhline(y=0, color='red', linestyle='--',
                   linewidth=1, alpha=0.5, label='τ(q) = 0')

        # Mark H position if estimated
        if self.H is not None:
            q_H = 1.0 / self.H
            ax1.axvline(x=q_H, color='green', linestyle='--',
                       linewidth=1, alpha=0.5, label=f'q = 1/H = {q_H:.3f}')
            ax1.plot([q_H], [0], 'go', markersize=10,
                    label=f'H = {self.H:.4f}')

        ax1.set_xlabel('q (moment order)', fontsize=12)
        ax1.set_ylabel('τ(q)', fontsize=12)
        ax1.set_title('Scaling Function τ(q)\n(Should be concave)',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right plot: Slopes (before subtracting 1)
        ax2.plot(self.q_values, self.slopes, 'o-',
                linewidth=2, markersize=6,
                color='darkorange', label='Slope = τ(q) + 1')

        # Add horizontal line at slope = 1 (where τ = 0)
        ax2.axhline(y=1, color='red', linestyle='--',
                   linewidth=1, alpha=0.5, label='Slope = 1 (τ = 0)')

        if self.H is not None:
            q_H = 1.0 / self.H
            ax2.axvline(x=q_H, color='green', linestyle='--',
                       linewidth=1, alpha=0.5)

        ax2.set_xlabel('q (moment order)', fontsize=12)
        ax2.set_ylabel('Slope', fontsize=12)
        ax2.set_title('Partition Plot Slopes\n(From Step 1)',
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_quality_metrics(self, save_path=None):
        """
        Plot R² values and other quality metrics from Step 1.

        This shows which q values have reliable τ(q) estimates.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot R² vs q
        ax.plot(self.q_values, self.r_squared, 'o-',
               linewidth=2, markersize=6, color='purple')

        # Add threshold line
        ax.axhline(y=config.MIN_R_SQUARED, color='red',
                  linestyle='--', linewidth=2,
                  label=f'Threshold = {config.MIN_R_SQUARED}')

        # Color-code points
        good_mask = self.r_squared >= config.MIN_R_SQUARED
        ax.scatter(self.q_values[good_mask], self.r_squared[good_mask],
                  s=100, c='green', alpha=0.5, zorder=5, label='Good fit')
        ax.scatter(self.q_values[~good_mask], self.r_squared[~good_mask],
                  s=100, c='red', alpha=0.5, zorder=5, label='Poor fit')

        ax.set_xlabel('q (moment order)', fontsize=12)
        ax.set_ylabel('R² (goodness of fit)', fontsize=12)
        ax.set_title('Quality of τ(q) Estimates\n(Higher R² = more reliable)',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Add summary text
        mean_r2 = self.r_squared.mean()
        good_pct = 100 * np.sum(good_mask) / len(good_mask)
        ax.text(0.02, 0.02,
               f'Mean R² = {mean_r2:.4f}\n{good_pct:.1f}% above threshold',
               transform=ax.transAxes,
               verticalalignment='bottom',
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def get_results_dataframe(self):
        """
        Get results as pandas DataFrame.

        Returns:
        --------
        pd.DataFrame
            DataFrame with q, slope, tau_q, r_squared
        """
        df = pd.DataFrame({
            'q': self.q_values,
            'slope': self.slopes,
            'tau_q': self.tau_q,
            'r_squared': self.r_squared
        })

        return df

    def save_results(self, filepath):
        """
        Save results to CSV.

        Parameters:
        -----------
        filepath : str
            Path to save CSV file
        """
        df = self.get_results_dataframe()

        # Add H as metadata in first row
        if self.H is not None:
            # Create metadata row
            metadata = pd.DataFrame({
                'q': ['H_parameter'],
                'slope': [self.H],
                'tau_q': [f'H={self.H:.6f}'],
                'r_squared': ['']
            })
            df = pd.concat([metadata, df], ignore_index=True)

        df.to_csv(filepath, index=False)

        if self.verbose:
            print(f"Results saved to: {filepath}")


def run_scaling_extraction(checker, output_dir=None, save_plots=True):
    """
    Complete workflow for extracting scaling function.

    Parameters:
    -----------
    checker : FractalityChecker
        Completed checker from Step 1
    output_dir : str, optional
        Directory to save results
    save_plots : bool
        Whether to save plots

    Returns:
    --------
    ScalingExtractor
        Extractor object with results
    """
    # Create output directory
    if output_dir is None:
        output_dir = config.PLOT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = ScalingExtractor(checker)

    # Run analysis
    print("\n" + "="*60)
    print("STEP 2: EXTRACTING SCALING FUNCTION τ(q)")
    print("="*60)

    # Estimate H
    H = extractor.estimate_H()

    # Validate
    extractor.validate_scaling_function()

    # CRITICAL: Check multifractality
    mf_results = extractor.check_multifractality()

    # Save results
    if config.SAVE_INTERMEDIATE:
        results_path = Path(output_dir) / "step2_scaling_results.csv"
        extractor.save_results(results_path)

    # Create plots
    if save_plots:
        scaling_plot_path = Path(output_dir) / "step2_scaling_function.png"
        extractor.plot_scaling_function(save_path=scaling_plot_path)

        quality_plot_path = Path(output_dir) / "step2_quality_metrics.png"
        extractor.plot_quality_metrics(save_path=quality_plot_path)

    # Print summary
    print("\n" + "="*60)
    print("STEP 2 COMPLETE")
    print("="*60)
    print(f"\nKey Results:")
    print(f"  H (self-affinity index) = {H:.4f}")
    print(f"  τ(q) range: [{extractor.tau_q.min():.4f}, {extractor.tau_q.max():.4f}]")
    print(f"  Mean R² from Step 1: {extractor.r_squared.mean():.4f}")
    print(f"  Multifractality Score: {mf_results['total_score']}/{mf_results['max_score']}")

    # Decision based on multifractality check
    print(f"\n" + "="*60)
    if mf_results['is_multifractal']:
        print("✓✓ DATA IS MULTIFRACTAL - PROCEED TO STEP 3")
        print("="*60)
        print(f"\n{mf_results['recommendation']}")
        print(f"\n→ Next: Run python run_step3.py")
    else:
        print("✗✗ DATA IS NOT SUFFICIENTLY MULTIFRACTAL")
        print("="*60)
        print(f"\n{mf_results['recommendation']}")
        print(f"\nConsider:")
        print(f"  - Using longer timeframe (15-min, hourly)")
        print(f"  - Different date range or asset")
        print(f"  - Comparing MMAR vs GARCH anyway (for research)")
        print(f"\nYou can still proceed to Step 3, but MMAR may not outperform simpler models.")

    print("="*60 + "\n")

    return extractor


if __name__ == "__main__":
    print("Step 2: Extract Scaling Function")
    print("="*60)
    print("\nThis script extracts τ(q) from Step 1 partition function results.")
    print("You must run Step 1 first to get the FractalityChecker object.\n")

    print("Usage:")
    print("  from step1_check_fractality import run_fractality_check")
    print("  from step2_extract_scaling import run_scaling_extraction")
    print("")
    print("  # Step 1")
    print("  checker = run_fractality_check(returns)")
    print("")
    print("  # Step 2")
    print("  extractor = run_scaling_extraction(checker)")
    print("")
    print("  # Access results")
    print("  H = extractor.H")
    print("  tau_q = extractor.tau_q")
