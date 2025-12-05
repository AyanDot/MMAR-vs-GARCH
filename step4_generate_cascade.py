"""
Step 4: Generate Multifractal Cascade (Trading Time)
Creates multifractal trading time θ(t) using multiplicative cascade

From paper Section 2.a and 4.d:
- Trading time θ(t) is the CDF of a multifractal measure
- Created via multiplicative cascade (iterative subdivision)
- Distribution type chosen from Step 3

Process:
1. Start with interval [0,1] with mass = 1
2. Divide into b subintervals (b=2 for binary cascade)
3. Sample multipliers from chosen distribution
4. Multiply parent mass by multipliers to get child masses
5. Repeat recursively k times
6. Integrate to get θ(t)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config


class CascadeGenerator:
    """
    Generates multifractal measures via multiplicative cascade.

    The cascade creates a complex, self-similar distribution of
    "trading time" that warps the FBM in Step 6.
    """

    def __init__(self, fitter, b=2, k=10, verbose=config.VERBOSE):
        """
        Initialize cascade generator.

        Parameters:
        -----------
        fitter : SpectrumFitter
            Completed fitter from Step 3
        b : int
            Branching factor (number of subintervals per division)
            Paper uses b=2
        k : int
            Number of cascade levels
            Paper suggests k=10 for good approximation
        verbose : bool
            Print detailed information
        """
        self.fitter = fitter
        self.b = b
        self.k = k
        self.verbose = verbose

        # Extract from previous steps
        self.H = fitter.H
        self.best_distribution = fitter.best_distribution
        self.best_params = fitter.best_params

        # Storage for cascade
        self.measure = None
        self.trading_time = None  # Integrated measure (CDF)

        # Final grid size
        self.n_points = b**k

        if self.verbose:
            print(f"\nCascade Generator Initialized")
            print(f"Distribution: {self.best_distribution}")
            print(f"Branching factor b = {b}")
            print(f"Cascade levels k = {k}")
            print(f"Grid points = {self.n_points}")

    def sample_multipliers(self, n_samples):
        """
        Sample multipliers from the chosen distribution.

        The multipliers M satisfy: V = -log_b(M)
        where V is distributed according to the fitted distribution.

        Parameters:
        -----------
        n_samples : int
            Number of multipliers to sample

        Returns:
        --------
        np.ndarray
            Array of multipliers
        """
        params = self.best_params

        if self.best_distribution == 'Normal':
            # V ~ N(μ, σ²)
            # μ = α₀/H
            # σ² = 2(α₀/H - 1) * ln(b)
            alpha_0 = params['alpha_0']
            mu = alpha_0 / self.H
            variance = 2 * (alpha_0 / self.H - 1) * np.log(self.b)

            if variance <= 0:
                variance = 0.01  # Fallback

            sigma = np.sqrt(variance)
            V = np.random.normal(mu, sigma, n_samples)

        elif self.best_distribution == 'Binomial':
            # V takes values α_min or α_max with equal probability
            alpha_min = params['alpha_min']
            alpha_max = params['alpha_max']
            V = np.random.choice([alpha_min, alpha_max], size=n_samples, p=[0.5, 0.5])

        elif self.best_distribution == 'Poisson':
            # V ~ Poisson(λ)
            # λ = α₀/H
            alpha_0 = params['alpha_0']
            lam = alpha_0 / self.H
            V = np.random.poisson(lam, n_samples)

        elif self.best_distribution == 'Gamma':
            # V ~ Gamma(shape=γ, rate=β)
            # β = ln(b) / (b^(1/γ) - 1)
            alpha_0 = params['alpha_0']
            gamma = params['gamma']

            # Calculate rate parameter
            beta = np.log(self.b) / (self.b**(1/gamma) - 1)

            V = np.random.gamma(gamma, 1/beta, n_samples)

        # Convert V to multipliers: M = b^(-V)
        M = self.b ** (-V)

        # Ensure multipliers are positive and finite
        M = np.clip(M, 1e-10, 1e10)

        # Normalize so they sum to b (conservation of mass)
        # For each group of b multipliers, they should sum to b
        M = M.reshape(-1, self.b)
        M = M / M.sum(axis=1, keepdims=True) * self.b
        M = M.flatten()

        return M

    def generate_cascade(self):
        """
        Generate multifractal measure via multiplicative cascade.

        Algorithm (from Figure 1 in paper):
        1. Start: measure[0] = 1
        2. For each level k:
           - Each interval is divided into b subintervals
           - Sample b multipliers for each interval
           - Child mass = parent mass × multiplier
        3. Result: fine-grained distribution of mass
        """
        if self.verbose:
            print("\nGenerating multiplicative cascade...")

        # Initialize: single interval with mass = 1
        measure = np.array([1.0])

        # Iterate through cascade levels
        for level in range(self.k):
            n_intervals = len(measure)
            new_measure = np.zeros(n_intervals * self.b)

            # Sample multipliers for all subdivisions
            multipliers = self.sample_multipliers(n_intervals * self.b)

            # Subdivide each interval
            for i in range(n_intervals):
                parent_mass = measure[i]
                for j in range(self.b):
                    child_idx = i * self.b + j
                    new_measure[child_idx] = parent_mass * multipliers[child_idx]

            measure = new_measure

            if self.verbose and (level % max(1, self.k // 5) == 0 or level == self.k - 1):
                print(f"  Level {level + 1}/{self.k}: {len(measure)} intervals")

        self.measure = measure

        # Normalize to unit mass
        self.measure = self.measure / self.measure.sum()

        if self.verbose:
            print(f"  ✓ Cascade complete: {len(self.measure)} points")
            print(f"  Mass range: [{self.measure.min():.2e}, {self.measure.max():.2e}]")

        return self.measure

    def integrate_measure(self):
        """
        Integrate measure to get trading time θ(t) (CDF).

        θ(t) is the cumulative distribution function of the measure.
        It represents the total trading time elapsed.

        Returns:
        --------
        np.ndarray
            Trading time array (CDF)
        """
        if self.measure is None:
            raise ValueError("Must run generate_cascade() first")

        # Cumulative sum (CDF)
        self.trading_time = np.cumsum(self.measure)

        # Ensure it ends at exactly 1.0
        self.trading_time = self.trading_time / self.trading_time[-1]

        if self.verbose:
            print("\n  ✓ Integrated measure → trading time θ(t)")
            print(f"  θ(t) range: [0, 1]")
            print(f"  Grid points: {len(self.trading_time)}")

        return self.trading_time

    def plot_cascade_results(self, save_path=None):
        """
        Visualize the multifractal measure and trading time.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Top left: Raw measure (mass distribution)
        ax = axes[0, 0]
        x = np.linspace(0, 1, len(self.measure))
        ax.plot(x, self.measure, 'b-', linewidth=0.5, alpha=0.7)
        ax.fill_between(x, self.measure, alpha=0.3)
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('Mass', fontsize=11)
        ax.set_title('Multifractal Measure\n(Mass Distribution)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Top right: Trading time (CDF)
        ax = axes[0, 1]
        x = np.linspace(0, 1, len(self.trading_time))
        ax.plot(x, self.trading_time, 'r-', linewidth=1.5)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Clock time (uniform)')
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('Cumulative Mass (θ)', fontsize=11)
        ax.set_title('Trading Time θ(t)\n(Integrated Measure)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Bottom left: Histogram of measure
        ax = axes[1, 0]
        ax.hist(self.measure, bins=50, color='blue', alpha=0.6, edgecolor='black')
        ax.set_xlabel('Mass', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Mass\n(Shows heterogeneity)', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Bottom right: Increments of trading time
        ax = axes[1, 1]
        increments = np.diff(self.trading_time)
        x = np.linspace(0, 1, len(increments))
        ax.plot(x, increments, 'g-', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Position', fontsize=11)
        ax.set_ylabel('dθ/dt', fontsize=11)
        ax.set_title('Trading Time Speed\n(dθ/dt shows volatility clustering)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def run_cascade_generation(fitter, output_dir=None, save_plots=True, b=2, k=10):
    """
    Complete workflow for generating multifractal cascade.

    Parameters:
    -----------
    fitter : SpectrumFitter
        Completed fitter from Step 3
    output_dir : str, optional
        Directory to save results
    save_plots : bool
        Whether to save plots
    b : int
        Branching factor
    k : int
        Number of cascade levels

    Returns:
    --------
    CascadeGenerator
        Generator object with trading time
    """
    # Create output directory
    if output_dir is None:
        output_dir = config.PLOT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = CascadeGenerator(fitter, b=b, k=k)

    # Run analysis
    print("\n" + "="*60)
    print("STEP 4: GENERATING MULTIFRACTAL CASCADE")
    print("="*60)

    # Generate cascade
    generator.generate_cascade()
    generator.integrate_measure()

    # Create plots
    if save_plots:
        cascade_plot_path = Path(output_dir) / "step4_cascade.png"
        generator.plot_cascade_results(save_path=cascade_plot_path)

    # Save trading time
    if config.SAVE_INTERMEDIATE:
        import pickle
        save_path = Path(config.OUTPUT_DIR) / "step4_generator.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(generator, f)
        if generator.verbose:
            print(f"\nTrading time saved to: {save_path}")

    # Print summary
    print("\n" + "="*60)
    print("STEP 4 COMPLETE")
    print("="*60)
    print(f"\nMultifractal trading time generated")
    print(f"Grid points: {len(generator.trading_time)}")
    print(f"Distribution: {generator.best_distribution}")

    print(f"\n→ Next: Run python run_step5.py")
    print("="*60 + "\n")

    return generator


if __name__ == "__main__":
    print("Step 4: Generate Multifractal Cascade")
    print("="*60)
    print("\nThis script generates trading time via multiplicative cascade.")
    print("You must run Steps 1-3 first.\n")
