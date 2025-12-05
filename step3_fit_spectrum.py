"""
Step 3: Fit Multifractal Spectrum f(α)
Transforms τ(q) to f(α) and fits to 4 distribution types

From paper Section 4.c:
- Transform τ(q) → f(α) using Legendre transform
- Fit to 4 distributions: Normal, Binomial, Poisson, Gamma
- Choose best fit (lowest squared error)

Legendre transform:
f_P(α) = αq - τ_P(q)
where α = dτ/dq
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from pathlib import Path
import config


class SpectrumFitter:
    """
    Fits multifractal spectrum f(α) from scaling function τ(q).

    The multifractal spectrum characterizes the distribution of
    local singularities (Hölder exponents) in the data.

    Process:
    1. Transform τ(q) → f(α) via Legendre transform
    2. Fit to 4 theoretical distributions
    3. Select best fit (minimum squared error)
    """

    def __init__(self, extractor, verbose=config.VERBOSE):
        """
        Initialize spectrum fitter.

        Parameters:
        -----------
        extractor : ScalingExtractor
            Completed extractor from Step 2 with τ(q) and H
        verbose : bool
            Print detailed information
        """
        self.extractor = extractor
        self.verbose = verbose

        # Extract from previous step
        self.q_values = extractor.q_values
        self.tau_q = extractor.tau_q
        self.H = extractor.H
        self.tau_q_func = extractor.tau_q_func

        # Storage for spectrum
        self.alpha_values = None
        self.f_alpha_values = None

        # Storage for fitted distributions
        self.fitted_distributions = {}
        self.best_distribution = None
        self.best_params = None

        if self.verbose:
            print(f"\nSpectrum Fitter Initialized")
            print(f"H = {self.H:.4f}")
            print(f"τ(q) range: [{self.tau_q.min():.4f}, {self.tau_q.max():.4f}]")

    def compute_multifractal_spectrum(self):
        """
        Compute f(α) from τ(q) using Legendre transform.

        Legendre transform:
        α(q) = dτ/dq
        f(α) = αq - τ(q)

        This transforms from q-space to α-space (singularity spectrum).
        """
        if self.verbose:
            print("\nComputing multifractal spectrum f(α)...")

        # Ensure interpolation function exists
        if self.tau_q_func is None:
            from scipy.interpolate import interp1d
            self.tau_q_func = interp1d(self.extractor.q_values,
                                       self.extractor.tau_q,
                                       kind='cubic',
                                       fill_value='extrapolate',
                                       bounds_error=False)

        # Compute α = dτ/dq numerically
        # Use central differences for interior points
        alpha_values = []
        f_alpha_values = []
        q_for_spectrum = []

        # Use the q_values from the extractor
        q_values = self.extractor.q_values
        tau_q = self.extractor.tau_q

        for i, q in enumerate(q_values):
            # Skip very small q values (numerical issues with derivative)
            if q < 0.5:  # Increased from 0.1 to avoid edge effects
                continue

            # Skip values too close to the edge
            if i == 0 or i == len(q_values) - 1:
                continue

            # Numerical derivative using finite difference
            try:
                # Calculate derivative: dτ/dq at point q
                # Use actual neighboring points instead of fixed h
                q_prev = q_values[i-1]
                q_next = q_values[i+1]
                tau_prev = tau_q[i-1]
                tau_next = tau_q[i+1]

                # Central difference
                alpha = (tau_next - tau_prev) / (q_next - q_prev)
                f_alpha = alpha * q - tau_q[i]

                # Only keep valid values
                if np.isfinite(alpha) and np.isfinite(f_alpha):
                    alpha_values.append(alpha)
                    f_alpha_values.append(f_alpha)
                    q_for_spectrum.append(q)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to compute spectrum at q={q:.2f}: {e}")
                continue

        self.alpha_values = np.array(alpha_values)
        self.f_alpha_values = np.array(f_alpha_values)
        self.q_for_spectrum = np.array(q_for_spectrum)

        if len(self.alpha_values) == 0:
            raise ValueError("Failed to compute any spectrum points! Check τ(q) data quality.")

        if self.verbose:
            print(f"  ✓ Computed {len(self.alpha_values)} spectrum points")
            print(f"  α range: [{self.alpha_values.min():.4f}, {self.alpha_values.max():.4f}]")
            print(f"  f(α) range: [{self.f_alpha_values.min():.4f}, {self.f_alpha_values.max():.4f}]")

            # Find peak (should be at f=1 theoretically)
            idx_max = np.argmax(self.f_alpha_values)
            alpha_0 = self.alpha_values[idx_max]
            f_max = self.f_alpha_values[idx_max]
            print(f"  Peak: α₀ = {alpha_0:.4f}, f(α₀) = {f_max:.4f}")

    def normal_spectrum(self, alpha, alpha_0, H):
        """
        Lognormal distribution spectrum (Appendix b.i).

        f_P(α) = 1 - (α - α₀)² / [4H(α₀ - H)]

        Parameters to fit: α₀ (peak location)
        """
        denominator = 4 * H * (alpha_0 - H)
        if denominator <= 0:
            return np.full_like(alpha, -np.inf)
        return 1 - (alpha - alpha_0)**2 / denominator

    def binomial_spectrum(self, alpha, alpha_min, alpha_max):
        """
        Binomial distribution spectrum (Appendix b.ii).

        f_P(α) = -(α_max - α)/(α_max - α_min) * log₂((α_max - α)/(α_max - α_min))
                 -(α - α_min)/(α_max - α_min) * log₂((α - α_min)/(α_max - α_min))

        Parameters to fit: α_min, α_max
        """
        if alpha_max <= alpha_min:
            return np.full_like(alpha, -np.inf)

        # Clip alpha to valid range
        alpha = np.clip(alpha, alpha_min + 1e-10, alpha_max - 1e-10)

        term1_ratio = (alpha_max - alpha) / (alpha_max - alpha_min)
        term2_ratio = (alpha - alpha_min) / (alpha_max - alpha_min)

        # Avoid log(0)
        term1_ratio = np.maximum(term1_ratio, 1e-10)
        term2_ratio = np.maximum(term2_ratio, 1e-10)

        f = -(term1_ratio * np.log2(term1_ratio) + term2_ratio * np.log2(term2_ratio))
        return f

    def poisson_spectrum(self, alpha, alpha_0, H, b=2):
        """
        Poisson distribution spectrum (Appendix b.iii).

        f_P(α) = 1 - α₀/(H·ln(b)) + (α/H)·log_b(α₀·e/α)

        Parameters to fit: α₀
        """
        if alpha_0 <= 0:
            return np.full_like(alpha, -np.inf)

        # Avoid division by zero
        alpha = np.maximum(alpha, 1e-10)

        term1 = 1 - alpha_0 / (H * np.log(b))
        term2 = (alpha / H) * (np.log(alpha_0 * np.e / alpha) / np.log(b))
        return term1 + term2

    def gamma_spectrum(self, alpha, alpha_0, gamma, b=2):
        """
        Gamma distribution spectrum (Appendix b.iv).

        f_P(α) = 1 + γ·log_b(α/α₀) + γ(α₀ - α)/(α₀·ln(b))

        Parameters to fit: α₀, γ
        """
        if alpha_0 <= 0 or gamma <= 0:
            return np.full_like(alpha, -np.inf)

        # Avoid division by zero
        alpha = np.maximum(alpha, 1e-10)

        term1 = gamma * np.log(alpha / alpha_0) / np.log(b)
        term2 = gamma * (alpha_0 - alpha) / (alpha_0 * np.log(b))
        return 1 + term1 + term2

    def fit_normal_distribution(self):
        """Fit lognormal spectrum."""
        # CRITICAL FIX: For lognormal spectrum, we MUST have alpha_0 > H
        # Otherwise denominator 4*H*(alpha_0 - H) becomes negative → NaN

        # Bounds: α₀ must be greater than H and within alpha range
        alpha_min_bound = max(self.H + 1e-6, self.alpha_values.min())
        alpha_max_bound = self.alpha_values.max()

        # Check if we have valid range
        if alpha_min_bound >= alpha_max_bound:
            # No valid range for lognormal fit
            if self.verbose:
                print(f"  Warning: Cannot fit Normal spectrum (need α₀ > H={self.H:.4f}, but alpha_max={alpha_max_bound:.4f})")
            return {'alpha_0': np.nan}, np.inf

        # Initial guess: middle of valid range
        alpha_0_init = (alpha_min_bound + alpha_max_bound) / 2

        def objective(params):
            alpha_0 = params[0]
            # Double-check constraint within objective
            if alpha_0 <= self.H:
                return 1e10
            f_fit = self.normal_spectrum(self.alpha_values, alpha_0, self.H)
            # Check for NaN/inf
            if not np.all(np.isfinite(f_fit)):
                return 1e10
            return np.sum((self.f_alpha_values - f_fit)**2)

        bounds = [(alpha_min_bound, alpha_max_bound)]

        result = minimize(objective, [alpha_0_init], bounds=bounds, method='L-BFGS-B')

        alpha_0_opt = result.x[0]
        sse = result.fun

        return {'alpha_0': alpha_0_opt}, sse

    def fit_binomial_distribution(self):
        """Fit binomial spectrum."""
        # Initial guess: use min and max of alpha
        alpha_min_init = self.alpha_values.min()
        alpha_max_init = self.alpha_values.max()

        def objective(params):
            alpha_min, alpha_max = params
            if alpha_max <= alpha_min:
                return 1e10
            f_fit = self.binomial_spectrum(self.alpha_values, alpha_min, alpha_max)
            return np.sum((self.f_alpha_values - f_fit)**2)

        # Bounds
        bounds = [(self.alpha_values.min() * 0.5, self.alpha_values.max() * 0.5),
                  (self.alpha_values.min() * 1.5, self.alpha_values.max() * 1.5)]

        result = minimize(objective, [alpha_min_init, alpha_max_init], bounds=bounds, method='L-BFGS-B')

        alpha_min_opt, alpha_max_opt = result.x
        sse = result.fun

        return {'alpha_min': alpha_min_opt, 'alpha_max': alpha_max_opt}, sse

    def fit_poisson_distribution(self):
        """Fit Poisson spectrum."""
        alpha_0_init = np.median(self.alpha_values)

        def objective(params):
            alpha_0 = params[0]
            f_fit = self.poisson_spectrum(self.alpha_values, alpha_0, self.H)
            return np.sum((self.f_alpha_values - f_fit)**2)

        bounds = [(self.alpha_values.min(), self.alpha_values.max())]

        result = minimize(objective, [alpha_0_init], bounds=bounds, method='L-BFGS-B')

        alpha_0_opt = result.x[0]
        sse = result.fun

        return {'alpha_0': alpha_0_opt}, sse

    def fit_gamma_distribution(self):
        """Fit Gamma spectrum."""
        alpha_0_init = np.median(self.alpha_values)
        gamma_init = 1.0

        def objective(params):
            alpha_0, gamma = params
            if gamma <= 0:
                return 1e10
            f_fit = self.gamma_spectrum(self.alpha_values, alpha_0, gamma)
            return np.sum((self.f_alpha_values - f_fit)**2)

        bounds = [(self.alpha_values.min(), self.alpha_values.max()),
                  (0.1, 10.0)]

        result = minimize(objective, [alpha_0_init, gamma_init], bounds=bounds, method='L-BFGS-B')

        alpha_0_opt, gamma_opt = result.x
        sse = result.fun

        return {'alpha_0': alpha_0_opt, 'gamma': gamma_opt}, sse

    def fit_all_distributions(self):
        """
        Fit all 4 distribution types and select best fit.

        Returns:
        --------
        str
            Name of best-fitting distribution
        """
        if self.verbose:
            print("\nFitting multifractal spectrum to 4 distributions...")

        # Fit each distribution
        distributions = {
            'Normal': self.fit_normal_distribution,
            'Binomial': self.fit_binomial_distribution,
            'Poisson': self.fit_poisson_distribution,
            'Gamma': self.fit_gamma_distribution
        }

        for name, fit_func in distributions.items():
            try:
                params, sse = fit_func()
                self.fitted_distributions[name] = {
                    'params': params,
                    'sse': sse,
                    'rmse': np.sqrt(sse / len(self.alpha_values))
                }
                if self.verbose:
                    print(f"  {name:10s}: SSE = {sse:.6f}, RMSE = {self.fitted_distributions[name]['rmse']:.6f}")
                    for key, val in params.items():
                        print(f"              {key} = {val:.6f}")
            except Exception as e:
                if self.verbose:
                    print(f"  {name:10s}: FAILED ({str(e)})")
                self.fitted_distributions[name] = None

        # Select best fit (lowest SSE)
        valid_dists = {k: v for k, v in self.fitted_distributions.items() if v is not None}
        if not valid_dists:
            raise ValueError("All distribution fits failed!")

        self.best_distribution = min(valid_dists.keys(), key=lambda k: valid_dists[k]['sse'])
        self.best_params = valid_dists[self.best_distribution]['params']

        if self.verbose:
            print(f"\n  ✓ BEST FIT: {self.best_distribution}")
            print(f"    RMSE = {valid_dists[self.best_distribution]['rmse']:.6f}")

        return self.best_distribution

    def generate_fitted_spectrum(self, alpha_grid):
        """
        Generate fitted spectrum values for plotting.

        Parameters:
        -----------
        alpha_grid : np.ndarray
            Grid of alpha values

        Returns:
        --------
        np.ndarray
            Fitted f(α) values
        """
        if self.best_distribution is None:
            raise ValueError("Must run fit_all_distributions() first")

        params = self.best_params

        if self.best_distribution == 'Normal':
            return self.normal_spectrum(alpha_grid, params['alpha_0'], self.H)
        elif self.best_distribution == 'Binomial':
            return self.binomial_spectrum(alpha_grid, params['alpha_min'], params['alpha_max'])
        elif self.best_distribution == 'Poisson':
            return self.poisson_spectrum(alpha_grid, params['alpha_0'], self.H)
        elif self.best_distribution == 'Gamma':
            return self.gamma_spectrum(alpha_grid, params['alpha_0'], params['gamma'])

    def plot_spectrum_fit(self, save_path=None):
        """Plot empirical spectrum and fitted distributions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Sample spectrum with all fits
        ax1.plot(self.alpha_values, self.f_alpha_values, 'ko',
                markersize=6, label='Empirical f(α)', alpha=0.6)

        # Generate smooth alpha grid for plotting
        alpha_grid = np.linspace(self.alpha_values.min() * 0.9,
                                self.alpha_values.max() * 1.1, 200)

        colors = {'Normal': 'blue', 'Binomial': 'red', 'Poisson': 'green', 'Gamma': 'orange'}

        for name, data in self.fitted_distributions.items():
            if data is None:
                continue

            params = data['params']

            if name == 'Normal':
                f_fit = self.normal_spectrum(alpha_grid, params['alpha_0'], self.H)
            elif name == 'Binomial':
                f_fit = self.binomial_spectrum(alpha_grid, params['alpha_min'], params['alpha_max'])
            elif name == 'Poisson':
                f_fit = self.poisson_spectrum(alpha_grid, params['alpha_0'], self.H)
            elif name == 'Gamma':
                f_fit = self.gamma_spectrum(alpha_grid, params['alpha_0'], params['gamma'])

            linewidth = 3 if name == self.best_distribution else 1
            linestyle = '-' if name == self.best_distribution else '--'
            label = f"{name} (RMSE={data['rmse']:.4f})"
            if name == self.best_distribution:
                label += " ★"

            ax1.plot(alpha_grid, f_fit, color=colors[name],
                    linewidth=linewidth, linestyle=linestyle, label=label, alpha=0.7)

        ax1.set_xlabel('α (Hölder exponent)', fontsize=12)
        ax1.set_ylabel('f(α)', fontsize=12)
        ax1.set_title('Multifractal Spectrum f(α)\nAll Distribution Fits',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.1])

        # Right: Best fit only
        ax2.plot(self.alpha_values, self.f_alpha_values, 'ko',
                markersize=8, label='Empirical f(α)', alpha=0.6)

        f_best = self.generate_fitted_spectrum(alpha_grid)
        ax2.plot(alpha_grid, f_best, 'r-', linewidth=2.5,
                label=f'{self.best_distribution} fit ★', alpha=0.8)

        ax2.set_xlabel('α (Hölder exponent)', fontsize=12)
        ax2.set_ylabel('f(α)', fontsize=12)
        ax2.set_title(f'Best Fit: {self.best_distribution} Distribution',
                     fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    def get_results_dataframe(self):
        """Get results as DataFrame."""
        spectrum_df = pd.DataFrame({
            'alpha': self.alpha_values,
            'f_alpha': self.f_alpha_values,
            'q': self.q_for_spectrum
        })

        return spectrum_df

    def save_results(self, filepath):
        """Save results to CSV."""
        df = self.get_results_dataframe()

        # Add metadata
        metadata = pd.DataFrame({
            'alpha': ['BEST_FIT', 'H_PARAMETER'],
            'f_alpha': [self.best_distribution, self.H],
            'q': ['', '']
        })

        # Add best fit parameters
        for key, val in self.best_params.items():
            metadata = pd.concat([metadata, pd.DataFrame({
                'alpha': [f'PARAM_{key}'],
                'f_alpha': [val],
                'q': ['']
            })], ignore_index=True)

        df = pd.concat([metadata, df], ignore_index=True)
        df.to_csv(filepath, index=False)

        if self.verbose:
            print(f"Results saved to: {filepath}")


def run_spectrum_fitting(extractor, output_dir=None, save_plots=True):
    """
    Complete workflow for fitting multifractal spectrum.

    Parameters:
    -----------
    extractor : ScalingExtractor
        Completed extractor from Step 2
    output_dir : str, optional
        Directory to save results
    save_plots : bool
        Whether to save plots

    Returns:
    --------
    SpectrumFitter
        Fitter object with results
    """
    # Create output directory
    if output_dir is None:
        output_dir = config.PLOT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize fitter
    fitter = SpectrumFitter(extractor)

    # Run analysis
    print("\n" + "="*60)
    print("STEP 3: FITTING MULTIFRACTAL SPECTRUM")
    print("="*60)

    # Compute spectrum
    fitter.compute_multifractal_spectrum()

    # Fit to distributions
    best_dist = fitter.fit_all_distributions()

    # Save results
    if config.SAVE_INTERMEDIATE:
        results_path = Path(output_dir) / "step3_spectrum_results.csv"
        fitter.save_results(results_path)

    # Create plots
    if save_plots:
        spectrum_plot_path = Path(output_dir) / "step3_spectrum_fit.png"
        fitter.plot_spectrum_fit(save_path=spectrum_plot_path)

    # Print summary
    print("\n" + "="*60)
    print("STEP 3 COMPLETE")
    print("="*60)
    print(f"\nBest-fitting distribution: {best_dist}")
    print(f"Parameters:")
    for key, val in fitter.best_params.items():
        print(f"  {key} = {val:.6f}")

    print(f"\n→ Next: Run python run_step4.py")
    print("="*60 + "\n")

    return fitter


if __name__ == "__main__":
    print("Step 3: Fit Multifractal Spectrum")
    print("="*60)
    print("\nThis script fits the multifractal spectrum f(α) to theoretical distributions.")
    print("You must run Steps 1 & 2 first.\n")
