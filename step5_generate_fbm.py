"""
Step 5: Generate Fractional Brownian Motion (FBM)
Creates FBM B_H(t) with self-affinity index H

From paper Section 2.a:
- FBM is a generalization of Brownian motion with long memory
- Characterized by Hurst exponent H
- H > 0.5: Persistent (trending)
- H = 0.5: Standard Brownian motion
- H < 0.5: Anti-persistent (mean-reverting)

Implementation uses Davies-Harte method (exact simulation)
"""

import numpy as np
from scipy.linalg import toeplitz, sqrtm
import matplotlib.pyplot as plt
from pathlib import Path
import config


class FBMGenerator:
    """
    Generates Fractional Brownian Motion using Davies-Harte method.

    FBM is a Gaussian process with stationary increments and
    long-range dependence controlled by the Hurst parameter H.
    """

    def __init__(self, H, n_points, verbose=config.VERBOSE):
        """
        Initialize FBM generator.

        Parameters:
        -----------
        H : float
            Hurst exponent (self-affinity index) from Step 2
        n_points : int
            Number of points to generate
        verbose : bool
            Print detailed information
        """
        self.H = H
        self.n_points = n_points
        self.verbose = verbose

        # Storage
        self.fbm = None
        self.fbm_increments = None

        if self.verbose:
            print(f"\nFBM Generator Initialized")
            print(f"H = {H:.4f}")
            print(f"Number of points = {n_points}")

            # Interpret H
            if abs(H - 0.5) < 0.05:
                print(f"  → H ≈ 0.5: Close to standard Brownian motion")
            elif H > 0.5:
                print(f"  → H > 0.5: Persistent (trending) behavior")
            else:
                print(f"  → H < 0.5: Anti-persistent (mean-reverting) behavior")

    def fbm_autocovariance(self, k):
        """
        Calculate autocovariance function of FBM increments.

        For FBM with Hurst parameter H:
        γ(k) = (1/2) * [|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H)]

        This is the covariance between increments separated by k steps.

        Parameters:
        -----------
        k : int or array
            Lag

        Returns:
        --------
        float or array
            Autocovariance at lag k
        """
        k = np.abs(k)
        return 0.5 * (np.abs(k + 1)**(2*self.H) - 2*np.abs(k)**(2*self.H) + np.abs(k - 1)**(2*self.H))

    def generate_fbm_davies_harte(self):
        """
        Generate FBM using Davies-Harte (exact) method.

        This method is exact for FBM and computationally efficient.

        Algorithm:
        1. Construct circulant covariance matrix from autocovariance
        2. Use FFT to generate correlated Gaussian increments
        3. Cumsum to get FBM path

        Returns:
        --------
        np.ndarray
            FBM path B_H(t)
        """
        if self.verbose:
            print("\nGenerating FBM using Davies-Harte method...")

        n = self.n_points

        # Build covariance vector for circulant embedding
        # First row of circulant matrix
        r = np.zeros(2 * n)
        for k in range(n):
            r[k] = self.fbm_autocovariance(k)

        # Mirror for circulant property
        for k in range(1, n):
            r[2*n - k] = r[k]

        # Eigenvalues via FFT
        lam = np.fft.fft(r).real

        # Check if all eigenvalues are non-negative (required for valid covariance)
        if np.any(lam < -1e-10):
            if self.verbose:
                print("  ⚠️  Negative eigenvalues detected, using Cholesky fallback")
            return self.generate_fbm_cholesky()

        # Ensure non-negative
        lam = np.maximum(lam, 0)

        # Generate complex Gaussian random variables
        # Real and imaginary parts are independent N(0,1)
        Z_real = np.random.randn(2*n)
        Z_imag = np.random.randn(2*n)
        Z = Z_real + 1j * Z_imag

        # Scale by sqrt of eigenvalues
        W = np.sqrt(lam / (2*n)) * Z

        # IFFT to get correlated increments
        w = np.fft.ifft(W)

        # Take real part and first n values
        increments = w[:n].real

        # Cumulative sum to get FBM
        self.fbm_increments = increments
        self.fbm = np.cumsum(increments)

        # Center at zero
        self.fbm = self.fbm - self.fbm[0]

        if self.verbose:
            print(f"  ✓ FBM generated: {len(self.fbm)} points")
            print(f"  FBM range: [{self.fbm.min():.4f}, {self.fbm.max():.4f}]")
            print(f"  Std dev: {np.std(self.fbm):.4f}")

        return self.fbm

    def generate_fbm_cholesky(self):
        """
        Fallback method using Cholesky decomposition.

        Slower but more stable for some parameter values.

        Returns:
        --------
        np.ndarray
            FBM path
        """
        if self.verbose:
            print("\nGenerating FBM using Cholesky method...")

        n = self.n_points

        # Build covariance matrix
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov_matrix[i, j] = self.fbm_autocovariance(i - j)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Add small diagonal for numerical stability
            cov_matrix += 1e-10 * np.eye(n)
            L = np.linalg.cholesky(cov_matrix)

        # Generate independent Gaussian random variables
        Z = np.random.randn(n)

        # Correlated increments
        increments = L @ Z

        # Cumulative sum
        self.fbm_increments = increments
        self.fbm = np.cumsum(increments)
        self.fbm = self.fbm - self.fbm[0]

        if self.verbose:
            print(f"  ✓ FBM generated: {len(self.fbm)} points")

        return self.fbm

    def scale_fbm(self, volatility):
        """
        Scale FBM by observed volatility.

        From paper Section 5.b:
        "The FBM is scaled by the sample standard deviation of 10 minute returns"

        Parameters:
        -----------
        volatility : float
            Sample standard deviation to match

        Returns:
        --------
        np.ndarray
            Scaled FBM
        """
        if self.fbm is None:
            raise ValueError("Must generate FBM first")

        # Current std dev
        current_std = np.std(self.fbm_increments)

        # Scale factor
        scale_factor = volatility / current_std if current_std > 0 else 1.0

        # Apply scaling
        self.fbm = self.fbm * scale_factor
        self.fbm_increments = self.fbm_increments * scale_factor

        if self.verbose:
            print(f"\nScaled FBM to match volatility = {volatility:.6f}")
            print(f"  Scale factor = {scale_factor:.4f}")

        return self.fbm

    def plot_fbm(self, save_path=None):
        """Visualize the generated FBM."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        t = np.arange(len(self.fbm))

        # Top left: FBM path
        ax = axes[0, 0]
        ax.plot(t, self.fbm, 'b-', linewidth=1, alpha=0.8)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('B_H(t)', fontsize=11)
        ax.set_title(f'Fractional Brownian Motion\n(H = {self.H:.4f})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Top right: FBM increments
        ax = axes[0, 1]
        ax.plot(t[:-1], self.fbm_increments[1:], 'r-', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('dB_H', fontsize=11)
        ax.set_title('FBM Increments\n(Returns)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Bottom left: Histogram of increments
        ax = axes[1, 0]
        ax.hist(self.fbm_increments, bins=50, color='green', alpha=0.6, edgecolor='black', density=True)
        ax.set_xlabel('Increment value', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Distribution of Increments\n(Should be Gaussian)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Overlay normal distribution
        x = np.linspace(self.fbm_increments.min(), self.fbm_increments.max(), 100)
        from scipy.stats import norm
        ax.plot(x, norm.pdf(x, np.mean(self.fbm_increments), np.std(self.fbm_increments)),
               'r-', linewidth=2, label='Normal fit')
        ax.legend(fontsize=9)

        # Bottom right: ACF of increments (should show long memory)
        ax = axes[1, 1]
        max_lag = min(100, len(self.fbm_increments) // 10)
        acf = np.correlate(self.fbm_increments - self.fbm_increments.mean(),
                          self.fbm_increments - self.fbm_increments.mean(),
                          mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]  # Normalize

        ax.plot(range(max_lag), acf[:max_lag], 'o-', markersize=3)
        ax.set_xlabel('Lag', fontsize=11)
        ax.set_ylabel('ACF', fontsize=11)
        ax.set_title('Autocorrelation Function\n(Long memory)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Add confidence interval
        conf_interval = 1.96 / np.sqrt(len(self.fbm_increments))
        ax.axhline(y=conf_interval, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-conf_interval, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def run_fbm_generation(extractor, n_points=None, output_dir=None, save_plots=True):
    """
    Complete workflow for generating FBM.

    Parameters:
    -----------
    extractor : ScalingExtractor
        Completed extractor from Step 2 (for H parameter)
    n_points : int, optional
        Number of points to generate (default: 2^10 = 1024)
    output_dir : str, optional
        Directory to save results
    save_plots : bool
        Whether to save plots

    Returns:
    --------
    FBMGenerator
        Generator object with FBM
    """
    # Default to 2^10 points
    if n_points is None:
        n_points = 2**10

    # Create output directory
    if output_dir is None:
        output_dir = config.PLOT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = FBMGenerator(H=extractor.H, n_points=n_points)

    # Run analysis
    print("\n" + "="*60)
    print("STEP 5: GENERATING FRACTIONAL BROWNIAN MOTION")
    print("="*60)

    # Generate FBM
    generator.generate_fbm_davies_harte()

    # Create plots
    if save_plots:
        fbm_plot_path = Path(output_dir) / "step5_fbm.png"
        generator.plot_fbm(save_path=fbm_plot_path)

    # Save FBM generator
    if config.SAVE_INTERMEDIATE:
        import pickle
        save_path = Path(config.OUTPUT_DIR) / "step5_fbm_generator.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(generator, f)
        if generator.verbose:
            print(f"\nFBM generator saved to: {save_path}")

    # Print summary
    print("\n" + "="*60)
    print("STEP 5 COMPLETE")
    print("="*60)
    print(f"\nFractional Brownian Motion generated")
    print(f"H = {generator.H:.4f}")
    print(f"Grid points: {len(generator.fbm)}")
    print(f"FBM std dev: {np.std(generator.fbm_increments):.6f}")

    print(f"\n→ Next: Run python run_step6.py")
    print("="*60 + "\n")

    return generator


if __name__ == "__main__":
    print("Step 5: Generate Fractional Brownian Motion")
    print("="*60)
    print("\nThis script generates FBM with Hurst parameter H from Step 2.")
    print("You must run Steps 1-2 first.\n")
