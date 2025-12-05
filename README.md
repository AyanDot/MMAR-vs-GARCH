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
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Key packages:
# - numpy, pandas, scipy, matplotlib
# - arch (for GARCH models)
# - nolds (for robust Hurst estimation)
# - MetaTrader5 (for live data)
```

### **Configuration**

Edit `config.py` to set your data parameters:

```python
SYMBOL = "EURUSD"
TIMEFRAME_MT5 = "M15"  # 15-minute bars
START_DATE = "2025-05-15"
END_DATE = "2025-07-01"
FORECAST_DAYS = 25
```

### **Run MMAR Model (Steps 1-7)**

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

### **Compare MMAR vs GARCH**

After completing all 7 MMAR steps:

```bash
# Run GARCH models and compare with MMAR
python run_garch_comparison.py

# This will:
# 1. Fit GARCH, EGARCH, and GJR-GARCH models
# 2. Generate forecasts using Zhang's methodology
# 3. Compare all models against realized volatility
# 4. Display winner (lowest forecast error)
```

### **Validate Forecasts**

```bash
# Compare forecasts with realized volatility
python run_comparison.py

# Shows:
# - Forecast accuracy
# - 95% confidence interval coverage
# - Forecast error metrics (RMSE, MAPE)
```

**Each step:**
1. Loads results from previous step
2. Runs its analysis
3. Saves results to `results/` directory
4. Creates diagnostic plots in `plots/` directory

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
├── 📄 requirements.txt             # Python dependencies
├── ⚙️  config.py                   # Configuration parameters
├── 🛠️  utils.py                    # Helper functions
├── 📥 data_loader.py               # MT5 integration + data loading
│
├── 📊 MMAR Steps (Steps 1-7)
│   ├── step1_check_fractality.py   # Partition function analysis
│   ├── step2_extract_scaling.py    # τ(q) extraction + H estimation
│   ├── step3_fit_spectrum.py       # Legendre transform + distribution fitting
│   ├── step4_generate_cascade.py   # Multiplicative cascade → trading time
│   ├── step5_generate_fbm.py       # Davies-Harte FBM generation
│   ├── step6_combine_model.py      # X(t) = B_H[θ(t)]
│   ├── step7_monte_carlo.py        # 10,000 Monte Carlo simulations
│   └── run_step[1-7].py            # Runner scripts for each step
│
├── 📊 GARCH Implementation
│   ├── garch_model.py              # GARCH, EGARCH, GJR-GARCH models
│   └── run_garch_comparison.py     # MMAR vs GARCH comparison script
│
├── 📊 Forecast Validation
│   ├── compare_forecast.py         # Compare forecast with realized volatility
│   └── run_comparison.py           # Validation runner script
│
├── 📁 data/                        # Historical price data (CSV)
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

This is a research implementation. Improvements to the technical implementation are always welcomed by the public and professionals. 

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

> *"Fractal geometry: expressing complex behavior in simple rules"* — Mandelbrot

---

**Built with ❤️ following Benoit Mandelbrot's legacy**

*Last updated: 2025*
