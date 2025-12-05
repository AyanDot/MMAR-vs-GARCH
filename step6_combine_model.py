"""
Step 6: Combine FBM with Trading Time
Creates MMAR: X(t) = B_H[θ(t)]

From paper equation (2.2):
X(t) = B_H[θ(t)]
where:
- B_H is FBM from Step 5
- θ(t) is multifractal trading time from Step 4

This "warps" the FBM by non-uniform trading time.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path
import config


class MMARCombiner:
    """
    Combines FBM and trading time to create the MMAR process.

    The combination warps the FBM by multifractal trading time,
    creating a process with:
    - Long memory (from FBM)
    - Volatility clustering (from trading time)
    """

    def __init__(self, fbm_generator, cascade_generator, verbose=config.VERBOSE):
        """
        Initialize MMAR combiner.

        Parameters:
        -----------
        fbm_generator : FBMGenerator
            Completed generator from Step 5
        cascade_generator : CascadeGenerator
            Completed generator from Step 4
        verbose : bool
            Print detailed information
        """
        self.fbm_gen = fbm_generator
        self.cascade_gen = cascade_generator
        self.verbose = verbose

        # Extract components
        self.fbm = fbm_generator.fbm
        self.trading_time = cascade_generator.trading_time

        # Storage
        self.mmar_process = None
        self.mmar_returns = None

        if self.verbose:
            print(f"\nMMAR Combiner Initialized")
            print(f"FBM points: {len(self.fbm)}")
            print(f"Trading time points: {len(self.trading_time)}")

    def combine_fbm_and_trading_time(self):
        """
        Compound FBM with trading time: X(t) = B_H[θ(t)].

        From paper Section 5.b:
        "For instance, if the value was 23.45, the relevant points
        of the FBM are the 24th and 25th entries. The price for this
        entry j of the series is a linear interpolation between the
        values of the FBM for the 24th and 25th entries."

        Algorithm:
        1. For each position j in trading time grid
        2. Get θ(j) (cumulative trading time)
        3. Map to FBM index: idx = θ(j) × len(FBM)
        4. Interpolate FBM at that fractional index
        5. This gives X(j) = B_H[θ(j)]

        Returns:
        --------
        np.ndarray
            MMAR process X(t)
        """
        if self.verbose:
            print("\nCombining FBM with trading time...")
            print("  X(t) = B_H[θ(t)]")

        n_points = len(self.trading_time)

        # Create interpolation function for FBM
        # Map [0, 1] (normalized time) to FBM values
        fbm_grid = np.linspace(0, 1, len(self.fbm))
        fbm_interp = interp1d(fbm_grid, self.fbm,
                             kind='linear',
                             bounds_error=False,
                             fill_value=(self.fbm[0], self.fbm[-1]))

        # For each point in trading time grid
        self.mmar_process = np.zeros(n_points)

        for j in range(n_points):
            # Get cumulative trading time at position j
            theta_j = self.trading_time[j]

            # Interpolate FBM at θ(j)
            self.mmar_process[j] = fbm_interp(theta_j)

        # Compute returns (log returns)
        self.mmar_returns = np.diff(self.mmar_process)

        if self.verbose:
            print(f"  ✓ MMAR process created: {len(self.mmar_process)} points")
            print(f"  Returns: {len(self.mmar_returns)} points")
            print(f"  Return std dev: {np.std(self.mmar_returns):.6f}")
            print(f"  Return range: [{self.mmar_returns.min():.4f}, {self.mmar_returns.max():.4f}]")

        return self.mmar_process, self.mmar_returns

    def plot_mmar_process(self, save_path=None):
        """Visualize the MMAR process and compare components."""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Top row: Components
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.fbm, 'b-', linewidth=1, alpha=0.7)
        ax1.set_title('FBM: B_H(t)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Clock time', fontsize=10)
        ax1.set_ylabel('B_H', fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        t_grid = np.linspace(0, 1, len(self.trading_time))
        ax2.plot(t_grid, self.trading_time, 'r-', linewidth=1.5)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Uniform time')
        ax2.set_title('Trading Time: θ(t)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Position', fontsize=10)
        ax2.set_ylabel('θ', fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Middle row: MMAR process
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(self.mmar_process, 'g-', linewidth=1, alpha=0.8)
        ax3.set_title('MMAR Process: X(t) = B_H[θ(t)]',
                     fontsize=13, fontweight='bold')
        ax3.set_xlabel('Time', fontsize=10)
        ax3.set_ylabel('X(t)', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # Bottom left: MMAR returns
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(self.mmar_returns, 'purple', linewidth=0.5, alpha=0.7)
        ax4.set_title('MMAR Returns: ΔX(t)',
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Time', fontsize=10)
        ax4.set_ylabel('Return', fontsize=10)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.grid(True, alpha=0.3)

        # Bottom right: Return distribution
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(self.mmar_returns, bins=50, color='orange', alpha=0.6, edgecolor='black', density=True)
        ax5.set_title('Return Distribution\n(Fat tails from multifractality)',
                     fontsize=12, fontweight='bold')
        ax5.set_xlabel('Return', fontsize=10)
        ax5.set_ylabel('Density', fontsize=10)
        ax5.grid(True, alpha=0.3)

        # Overlay normal for comparison
        from scipy.stats import norm
        x = np.linspace(self.mmar_returns.min(), self.mmar_returns.max(), 100)
        ax5.plot(x, norm.pdf(x, np.mean(self.mmar_returns), np.std(self.mmar_returns)),
                'r--', linewidth=2, label='Normal', alpha=0.7)
        ax5.legend(fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def run_mmar_combination(fbm_generator, cascade_generator, output_dir=None, save_plots=True):
    """
    Complete workflow for combining FBM and trading time.

    Parameters:
    -----------
    fbm_generator : FBMGenerator
        Completed generator from Step 5
    cascade_generator : CascadeGenerator
        Completed generator from Step 4
    output_dir : str, optional
        Directory to save results
    save_plots : bool
        Whether to save plots

    Returns:
    --------
    MMARCombiner
        Combiner object with MMAR process
    """
    # Create output directory
    if output_dir is None:
        output_dir = config.PLOT_DIR

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize combiner
    combiner = MMARCombiner(fbm_generator, cascade_generator)

    # Run analysis
    print("\n" + "="*60)
    print("STEP 6: COMBINING FBM WITH TRADING TIME")
    print("="*60)

    # Combine
    combiner.combine_fbm_and_trading_time()

    # Create plots
    if save_plots:
        mmar_plot_path = Path(output_dir) / "step6_mmar_process.png"
        combiner.plot_mmar_process(save_path=mmar_plot_path)

    # Save combiner
    if config.SAVE_INTERMEDIATE:
        import pickle
        save_path = Path(config.OUTPUT_DIR) / "step6_combiner.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(combiner, f)
        if combiner.verbose:
            print(f"\nMMAR combiner saved to: {save_path}")

    # Print summary
    print("\n" + "="*60)
    print("STEP 6 COMPLETE")
    print("="*60)
    print(f"\nMMAR process X(t) = B_H[θ(t)] created")
    print(f"Process points: {len(combiner.mmar_process)}")
    print(f"Return volatility: {np.std(combiner.mmar_returns):.6f}")

    print(f"\n→ Next: Run python run_step7.py")
    print("="*60 + "\n")

    return combiner


if __name__ == "__main__":
    print("Step 6: Combine FBM with Trading Time")
    print("="*60)
    print("\nThis script combines FBM and trading time to create MMAR.")
    print("You must run Steps 1-5 first.\n")
