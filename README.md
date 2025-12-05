# MMAR Volatility Forecasting

**Multifractal Model of Asset Returns (MMAR)** implementation for volatility forecasting, based on the groundbreaking work of Benoit Mandelbrot.

> *"The study of roughness, of the irregular and jagged"* — Benoit B. Mandelbrot

---

## 📄 Paper Reference

**"Volatility Forecasting with the Multifractal Model of Asset Returns"**
*Yidong (Terrence) Zhang, University of Florida, 2017*

This implementation follows Zhang's empirical application of Mandelbrot's MMAR to intraday stock prices, demonstrating that **MMAR significantly outperforms GARCH models** for volatility forecasting.

---

## 🎯 What is MMAR?

The MMAR models financial returns as a **Fractional Brownian Motion (FBM) compounded by multifractal trading time**:

```
X(t) = B_H[θ(t)]
```

Where:
- **B_H(t)** = Fractional Brownian Motion with Hurst exponent H
  - Captures **long memory** and persistence in returns
  - H > 0.5: trending behavior
  - H = 0.5: random walk
  - H < 0.5: mean-reverting behavior

- **θ(t)** = Multifractal trading time (CDF of multifractal measure)
  - Captures **volatility clustering** and fat tails
  - Created via multiplicative cascade
  - Represents non-uniform "speed" of market time

This combination produces a process with:
1. ✅ **Long memory** (from FBM)
2. ✅ **Fat tails** (from multifractal time)
3. ✅ **Volatility clustering** (from multifractal time)
4. ✅ **Scale consistency** (moment scaling)

---

## 🔬 The 7-Step Construction Process

### **Step 1: Check Fractality**
**Does moment scaling exist?**

Use the partition function to test for multifractal behavior:

```
S_q(T, Δt) = Σ |r(iΔt, Δt)|^q
```

Plot log₁₀(S_q) vs log₁₀(Δt) for various q values.

**✓ Linearity = Moment Scaling = Fractality Exists**

**Key Parameters:**
- q range: 0.01 to 30.0 (moment orders)
- Δt range: 1 to 150 observations (time scales)
- R² threshold: ≥ 0.60 (paper achieved 0.66)

**Critical Implementation:**
- Uses **NON-OVERLAPPING intervals** (Zhang's methodology)
- Avoids artificial correlation from overlapping windows

---

### **Step 2: Extract Scaling Function τ(q)**
**How fractal is it?**

Extract the scaling function from partition plot slopes:

```
τ(q) = slope - 1
```

Then estimate the **Hurst exponent H** where τ(1/H) = 0.

**Implementation:**
- **Primary method:** `nolds.hurst_rs()` — robust R/S (rescaled range) analysis
- **Fallback method:** Find where τ(1/H) = 0 numerically

**Why nolds?**
- More robust than numerical root-finding
- Handles edge cases better
- Industry-standard implementation

**Critical Test: Multifractality Check**

3 independent tests (scored 0-9 total):
1. **Concavity** — Is τ(q) concave? (Linear = monofractal = MMAR fails)
2. **Linearity** — How different from random walk?
3. **Slope Variation** — Do local slopes vary?

**Verdict:**
- Score ≥ 7: **Strongly multifractal** → MMAR highly recommended
- Score ≥ 5: **Moderately multifractal** → MMAR worth trying
- Score < 5: **Not multifractal** → MMAR not recommended (use GARCH instead)

---

### **Step 3: Fit Multifractal Spectrum f(α)**
**What type of fractal?**

Transform τ(q) → f(α) using **Legendre transform**:

```
α = dτ/dq
f(α) = αq - τ(q)
```

Fit to **4 theoretical distributions**:
1. **Lognormal**: `f(α) = 1 - (α - α₀)² / [4H(α₀ - H)]`
2. **Binomial**: Discrete two-point distribution
3. **Poisson**: Counting process
4. **Gamma**: Continuous skewed distribution

**Selection:** Choose distribution with **lowest squared error**

This determines the **type of multifractal measure** used in Step 4.

---

### **Step 4: Generate Multifractal Cascade**
**Create trading time θ(t)**

Build multifractal measure via **multiplicative cascade**:

```
Algorithm:
1. Start: [0, 1] with mass = 1
2. Divide into b=2 subintervals
3. Sample multipliers M from chosen distribution
4. Child mass = parent mass × M
5. Repeat k=10 levels → 2^10 = 1024 intervals
6. Integrate to get θ(t) (CDF)
```

**Result:** Trading time θ(t) with:
- Peaks = high volatility periods (fast trading time)
- Valleys = low volatility periods (slow trading time)

**Key insight:** Unlike clock time (uniform), trading time varies — capturing volatility clustering!

---

### **Step 5: Generate Fractional Brownian Motion**
**Create long-memory process B_H(t)**

Generate FBM using **Davies-Harte method** (exact simulation):

```python
# Autocovariance of FBM increments
γ(k) = 0.5 × [|k+1|^(2H) - 2|k|^(2H) + |k-1|^(2H)]
```

**Algorithm:**
1. Construct circulant covariance matrix
2. Use FFT to get eigenvalues
3. Generate complex Gaussian random variables
4. Scale by sqrt(eigenvalues)
5. IFFT to get correlated increments
6. Cumsum to get FBM path

**Fallback:** Cholesky decomposition (slower but stable)

**Scaling:** FBM is scaled by sample volatility from historical data

---

### **Step 6: Combine FBM with Trading Time**
**Warp FBM by multifractal time**

The MMAR equation comes to life:

```
X(j) = B_H[θ(j)]
```

**Algorithm:**
1. For each position j in trading time grid:
2. Get θ(j) = cumulative trading time
3. Map to FBM index: idx = θ(j) × len(FBM)
4. **Linear interpolate** FBM at fractional index
5. Result: X(j) = B_H[θ(j)]

**Key insight:** When θ(t) speeds up (volatility clustering), we sample FBM faster → larger returns!

---

### **Step 7: Monte Carlo Forecasting**
**Run 10,000 simulations**

```python
for i in range(10,000):
    # Generate fresh cascade (Step 4)
    # Generate fresh FBM (Step 5)
    # Combine (Step 6)
    # Calculate volatility
    volatility[i] = std(returns)

forecast = mean(volatility)
```

**Output:**
- **Point estimate:** Mean volatility
- **Uncertainty:** Std dev of volatility
- **95% CI:** [mean - 1.96×std, mean + 1.96×std]

**Why 10,000?** Ensures stable convergence (paper standard)

---

## 🚀 Quick Start

### **Installation**

```bash
# Install dependencies
pip install -r requirements.txt

# Key packages:
# - numpy, pandas, scipy, matplotlib
# - nolds (for robust Hurst estimation)
# - MetaTrader5 (for live data)
```

### **Run All 7 Steps**

```bash
# Step 1: Check fractality (moment scaling)
python run_step1.py

# Step 2: Extract τ(q) and estimate H
python run_step2.py

# Step 3: Fit multifractal spectrum
python run_step3.py

# Step 4: Generate multifractal cascade
python run_step4.py

# Step 5: Generate FBM
python run_step5.py

# Step 6: Combine FBM + trading time
python run_step6.py

# Step 7: Monte Carlo forecast (10,000 sims)
python run_step7.py
```

**Each step:**
1. Loads results from previous step
2. Runs its analysis
3. Saves results for next step
4. Creates diagnostic plots

---

## 📊 Understanding Your Forecast

### **Example Output:**

```
Volatility Forecast:
  Point estimate: 0.000820
  Uncertainty (std dev): 0.000010
  95% Confidence Interval: [0.000801, 0.000840]
```

### **What Does This Mean?**

**For 5-minute EURUSD data:**

| Timeframe | Volatility | Formula |
|-----------|-----------|---------|
| **5-min** | 0.082% | 0.000820 × 100% |
| **Hourly** | 0.28% | 0.000820 × √12 |
| **Daily** | 1.39% | 0.000820 × √288 |
| **Annual** | 22.1% | 0.000820 × √72,576 |

**Is 22% annual volatility realistic for EURUSD?**

✅ **YES!**
- Calm markets: 8-12%
- Normal markets: 12-18%
- Volatile markets: 18-25%

Your forecast of 22% suggests elevated volatility.

---

## 📈 Practical Applications

### **1. Position Sizing**

```python
# Risk-adjusted position sizing
account_balance = 10000
risk_per_trade = 0.01  # 1%
vol_forecast = 0.000820

position_size = (account_balance * risk_per_trade) / vol_forecast
# Smaller positions when volatility is high!
```

### **2. Stop Loss Placement**

```python
# 2-sigma stop (95% confidence)
holding_periods = 12  # 1 hour = 12 × 5-min

stop_distance = 2 * vol_forecast * sqrt(holding_periods)
# Example: 2 × 0.000820 × √12 ≈ 0.0057 (57 pips)
```

### **3. Expected Price Range**

```python
# Next 24 hours
expected_range = vol_forecast * sqrt(288)  # 288 × 5-min in 24h
# ≈ 0.0139 (139 pips daily range)
```

---

## 📁 Project Structure

```
MMAR_Forecast/
│
├── 📄 README.md                    # This file
├── ⚙️  config.py                   # Configuration parameters
├── 🛠️  utils.py                    # Helper functions
├── 📥 data_loader.py               # MT5 integration + data loading
│
├── 📊 Step 1: Check Fractality
│   ├── step1_check_fractality.py   # Partition function analysis
│   └── run_step1.py                # Runner script
│
├── 📊 Step 2: Extract Scaling
│   ├── step2_extract_scaling.py    # τ(q) extraction + H estimation
│   └── run_step2.py                # Runner script
│
├── 📊 Step 3: Fit Spectrum
│   ├── step3_fit_spectrum.py       # Legendre transform + distribution fitting
│   └── run_step3.py                # Runner script
│
├── 📊 Step 4: Generate Cascade
│   ├── step4_generate_cascade.py   # Multiplicative cascade → trading time
│   └── run_step4.py                # Runner script
│
├── 📊 Step 5: Generate FBM
│   ├── step5_generate_fbm.py       # Davies-Harte FBM generation
│   └── run_step5.py                # Runner script
│
├── 📊 Step 6: Combine Model
│   ├── step6_combine_model.py      # X(t) = B_H[θ(t)]
│   └── run_step6.py                # Runner script
│
├── 📊 Step 7: Monte Carlo
│   ├── step7_monte_carlo.py        # 10,000 simulations
│   └── run_step7.py                # Runner script
│
├── 📁 data/                        # Historical price data
├── 📁 results/                     # Pickle files (intermediate results)
├── 📁 plots/                       # PNG diagnostic plots
│
└── 📑 Volatility Forecasting with the MMAR.pdf  # Original paper
```

---

## 🔧 Configuration

Edit `config.py` to customize:

### **Data Parameters**

```python
SYMBOL = "EURUSD"
TIMEFRAME_MT5 = "M5"  # 5-minute bars
START_DATE = "2021-01-01"
END_DATE = "2025-07-01"
```

### **Partition Function (Step 1)**

```python
DELTA_T_MIN = 1       # Min: 1 observation
DELTA_T_MAX = 150     # Max: 150 observations (12.5 hours for M5)
Q_MIN = 0.01
Q_MAX = 30.0
MIN_R_SQUARED = 0.60  # Fractality threshold
```

### **Monte Carlo (Step 7)**

```python
NUM_SIMULATIONS = 10000  # Paper standard
RANDOM_SEED = 42         # For reproducibility
```

### **Output Directories**

```python
OUTPUT_DIR = ".../results"  # Pickle files
PLOT_DIR = ".../plots"      # PNG plots
DATA_DIR = ".../data"       # CSV data
```

---

## 📚 Key Concepts

### **Moment Scaling**

The defining property of multifractals:

```
E(|X(t)|^q) = c(q) × t^(τ(q)+1)
```

If this holds across multiple time scales → data is multifractal.

### **Self-Similarity**

Zooming in/out reveals similar patterns:
- 5-min chart looks like hourly chart
- Daily chart looks like weekly chart
- This is **scale invariance**

### **Trading Time**

> *"Reliance upon a single time scale leads to inefficiency"* — Mandelbrot

Instead of uniform clock time, MMAR uses **multifractal time**:
- Fast during volatile periods
- Slow during calm periods
- Captures reality: markets don't move uniformly!

### **Why MMAR Outperforms GARCH**

| Feature | GARCH | MMAR |
|---------|-------|------|
| Long memory | ❌ (except FIGARCH) | ✅ |
| Fat tails | ✅ | ✅ |
| Volatility clustering | ✅ | ✅ |
| Scale consistency | ❌ | ✅ |
| Multiple time scales | ❌ | ✅ |

**Paper result:** MMAR had **significantly lower RMSE** than GARCH variants.

---

## ⚠️ Important Implementation Notes

### **1. Non-Overlapping Intervals (Step 1)**

✅ **Correct:**
```python
# Divide returns into N = len(returns) // delta_t chunks
# Each chunk is independent
```

❌ **Wrong:**
```python
# Sliding window → artificial correlation
```

### **2. Hurst Estimation (Step 2)**

✅ **Use nolds.hurst_rs()** (robust R/S method)

❌ **Avoid:** Finding τ(1/H) = 0 numerically (unstable)

Typical values:
- FX markets: H = 0.48-0.55
- Stock markets: H = 0.52-0.58
- H = 0.1 → **ERROR**, re-estimate!

### **3. Multifractality Test (Step 2)**

**Critical:** If τ(q) is linear → data is **monofractal** → MMAR will fail!

Check concavity before proceeding to Step 3.

### **4. Spectrum Fitting (Step 3)**

Use **all 4 distributions**, pick best fit by squared error.

Most common: Lognormal or Gamma

### **5. Monte Carlo (Step 7)**

10,000 simulations = **10-30 minutes** runtime.

For testing: use 100 simulations first.

---

## 🎓 Theory Deep Dive

### **Fractional Brownian Motion (FBM)**

Standard Brownian motion has:
- Independent increments
- H = 0.5 (no memory)

FBM generalizes this:
- **Dependent** increments
- **H ≠ 0.5** (long memory)

**Autocovariance:**
```
Cov[dB_H(s), dB_H(t)] ∝ |t-s|^(2H-2)
```

H > 0.5: Positive autocorrelation (persistence)
H < 0.5: Negative autocorrelation (anti-persistence)

### **Multiplicative Cascade**

Simple rule → complex pattern:

```
Step k=0: [1]
Step k=1: [0.6, 0.4]           (split, multiply by M₁, M₂)
Step k=2: [0.3, 0.3, 0.2, 0.2] (split each, multiply)
...
```

After k=10 steps: 1024 intervals with complex mass distribution.

**Integral = θ(t) = multifractal time**

### **Legendre Transform**

Converts scaling function τ(q) to multifractal spectrum f(α):

```
α = dτ/dq              (Hölder exponent)
f(α) = αq - τ(q)       (dimension function)
```

**Interpretation:**
- α: Local roughness exponent
- f(α): Frequency of points with roughness α

Inverted parabola shape → multifractal

---

## 🔬 Validation & Testing

### **Step 1: In-Sample Validation**

Compare forecast with realized volatility:

```python
# After Step 7
forecast = 0.000820

# Calculate realized volatility from actual data
realized = np.std(actual_returns)

error = abs(forecast - realized)
```

### **Step 2: GARCH Comparison**

Implement GARCH(1,1) for same data:

```python
from arch import arch_model

model = arch_model(returns, vol='Garch', p=1, q=1)
result = model.fit()
garch_forecast = result.forecast(horizon=288).variance[-1]
```

**Expected:** MMAR < GARCH (lower error)

### **Step 3: Out-of-Sample Testing**

```python
# Train on Jan-Jun 2024
# Test on Jul-Dec 2024

# Rolling window:
for month in range(7, 13):
    train_data = data[:month]
    test_data = data[month]

    # Run Steps 1-7 on train_data
    # Compare forecast with test_data
```

---

## 📖 Further Reading

### **Foundational Papers**

1. **Mandelbrot, B. B., Calvet, L., Fisher, A. (1997)**
   *"The Multifractal Model of Asset Returns"*
   Cowles Foundation Discussion Paper No. 1164

2. **Calvet, L., Fisher, A. (2002)**
   *"Multifractality in Asset Returns: Theory and Evidence"*
   Review of Economics and Statistics, 84(3), 381-406

3. **Zhang, Y. T. (2017)**
   *"Volatility Forecasting with the Multifractal Model of Asset Returns"*
   University of Florida (this implementation)

### **Related Concepts**

- **Hurst, H. E. (1951)** — Long-term storage capacity of reservoirs
- **Mandelbrot, B. B. (1963)** — Fat tails in asset returns
- **Mandelbrot & Ness (1968)** — Fractional Brownian Motion
- **Engle (1982), Bollerslev (1986)** — GARCH models

### **Books**

- **Mandelbrot, B. B. (2006)**
  *"The (Mis)behavior of Markets: A Fractal View of Financial Turbulence"*

---

## 🤝 Contributing

This is a research implementation. Improvements welcome:

1. **Additional distributions** for Step 3
2. **Faster cascade generation** (GPU acceleration)
3. **Real-time forecasting** integration
4. **GARCH benchmark** comparison
5. **Walk-forward validation** framework

---

## ⚖️ License & Disclaimer

**Educational/Research Use**

This code implements academic research. Use at your own risk.

**Not financial advice.** Past performance ≠ future results.

---

## 🙏 Acknowledgments

- **Benoit B. Mandelbrot** — Father of fractal geometry
- **Yidong (Terrence) Zhang** — Empirical MMAR implementation
- **Laurent Calvet & Adlai Fisher** — MMAR theory development

---

## 📞 Support

For issues with:
- **Installation:** Check `requirements.txt`
- **MT5 connection:** Verify `config.py` settings
- **Low R² in Step 1:** Use more data or lower `MIN_R_SQUARED`
- **Extreme H values:** Install `nolds` and re-run Step 2
- **No spectrum points:** Check τ(q) quality in Step 2

---

## 🎯 Summary

**MMAR captures what GARCH misses:**

- ✅ Markets have memory (H ≠ 0.5)
- ✅ Time is not uniform (multifractal θ)
- ✅ Volatility clusters in complex ways
- ✅ Same patterns across all time scales

**Result:** More accurate volatility forecasts for risk management and position sizing.

> *"Fractal geometry: expressing complex behavior in simple rules"* — Mandelbrot

---

**Built with ❤️ following Benoit Mandelbrot's legacy**

*Last updated: 2025*
