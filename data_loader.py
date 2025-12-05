"""
Data Loader for MMAR Volatility Forecasting
Loads and preprocesses data from MetaTrader 5 or CSV files for multifractal analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import config

# Try to import MetaTrader5
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("Warning: MetaTrader5 module not installed. Only CSV loading available.")
    print("Install with: pip install MetaTrader5")


class DataLoader:
    """
    Load and preprocess price data for MMAR analysis.

    Responsibilities:
    - Fetch data from MetaTrader 5 terminal (primary method)
    - Load data from CSV files (fallback method)
    - Calculate log returns
    - Handle missing data
    - Prepare data in format required for partition function
    """

    def __init__(self, symbol=config.SYMBOL, start_date=config.START_DATE,
                 end_date=config.END_DATE, verbose=config.VERBOSE):
        """
        Initialize data loader.

        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'EURUSD')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        verbose : bool
            Print detailed information
        """
        self.symbol = symbol
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.verbose = verbose

        self.data = None
        self.returns = None
        self.prices = None
        self.mt5_initialized = False

    def _initialize_mt5(self):
        """
        Initialize MetaTrader 5 connection.

        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if not MT5_AVAILABLE:
            if self.verbose:
                print("MetaTrader5 module not available")
            return False

        if not mt5.initialize():
            if self.verbose:
                print(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        self.mt5_initialized = True

        if self.verbose:
            terminal_info = mt5.terminal_info()
            account_info = mt5.account_info()
            print(f"\n✓ MetaTrader 5 Connected")
            print(f"  Terminal: {terminal_info.company}")
            print(f"  Account: {account_info.login if account_info else 'N/A'}")
            print(f"  Server: {account_info.server if account_info else 'N/A'}")

        return True

    def _shutdown_mt5(self):
        """Shutdown MT5 connection."""
        if self.mt5_initialized and MT5_AVAILABLE:
            mt5.shutdown()
            self.mt5_initialized = False

    def _get_mt5_timeframe(self, timeframe_str):
        """
        Convert timeframe string to MT5 constant.

        Parameters:
        -----------
        timeframe_str : str
            Timeframe string (e.g., 'M1', 'M5', 'H1', 'D1')

        Returns:
        --------
        int
            MT5 timeframe constant
        """
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1,
        }

        return timeframe_map.get(timeframe_str.upper(), mt5.TIMEFRAME_M1)

    def load_from_mt5(self, symbol=None, timeframe=None, start_date=None, end_date=None):
        """
        Load data directly from MetaTrader 5 terminal.

        This is the PRIMARY and RECOMMENDED method for loading data.

        Parameters:
        -----------
        symbol : str, optional
            Trading symbol (uses config.SYMBOL if None)
        timeframe : str, optional
            Timeframe (uses config.TIMEFRAME_MT5 if None)
        start_date : str, optional
            Start date (uses config.START_DATE if None)
        end_date : str, optional
            End date (uses config.END_DATE if None)

        Returns:
        --------
        pd.DataFrame
            Loaded price data with OHLCV
        """
        # Use defaults if not provided
        symbol = symbol or self.symbol
        timeframe = timeframe or config.TIMEFRAME_MT5
        start_date = pd.to_datetime(start_date or self.start_date)
        end_date = pd.to_datetime(end_date or self.end_date)

        if self.verbose:
            print(f"\nLoading data from MetaTrader 5...")
            print(f"  Symbol: {symbol}")
            print(f"  Timeframe: {timeframe}")
            print(f"  From: {start_date.date()}")
            print(f"  To: {end_date.date()}")

        # Initialize MT5
        if not self._initialize_mt5():
            raise RuntimeError("Failed to initialize MetaTrader 5. Is the terminal running?")

        # Get MT5 timeframe constant
        mt5_tf = self._get_mt5_timeframe(timeframe)

        # Fetch rates
        rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)

        if rates is None or len(rates) == 0:
            error_msg = f"Failed to fetch data: {mt5.last_error()}"
            if self.verbose:
                print(f"✗ {error_msg}")
            raise ValueError(error_msg)

        # Convert to DataFrame
        df = pd.DataFrame(rates)

        # Convert time to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Rename columns to standard format
        df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']

        if self.verbose:
            print(f"✓ Loaded {len(df)} bars")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            print(f"  First close: {df['close'].iloc[0]:.5f}")
            print(f"  Last close: {df['close'].iloc[-1]:.5f}")

        self.data = df
        return df

    def load_csv(self, filepath):
        """
        Load price data from CSV file (fallback method).

        Expected CSV format:
        - Columns: datetime, open, high, low, close, volume
        - datetime format: YYYY-MM-DD HH:MM:SS

        Parameters:
        -----------
        filepath : str
            Path to CSV file

        Returns:
        --------
        pd.DataFrame
            Loaded price data
        """
        if self.verbose:
            print(f"\nLoading data from CSV: {filepath}")

        # Load CSV
        df = pd.read_csv(filepath)

        # Parse datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'])
        elif 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        else:
            # Try to parse index
            df['datetime'] = pd.to_datetime(df.index)

        # Set datetime as index
        df.set_index('datetime', inplace=True)

        # Standardize column names
        df.columns = df.columns.str.lower()

        # Filter date range
        df = df.loc[self.start_date:self.end_date]

        if self.verbose:
            print(f"✓ Loaded {len(df)} rows")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")

        self.data = df
        return df

    def calculate_returns(self, price_column='close', method='log'):
        """
        Calculate returns from price data.

        Parameters:
        -----------
        price_column : str
            Column name for prices (default: 'close')
        method : str
            'log' for log returns (default, recommended), 'simple' for simple returns

        Returns:
        --------
        pd.Series
            Calculated returns
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_from_mt5() or load_csv() first.")

        # Get prices
        prices = self.data[price_column].values
        self.prices = prices

        if method == 'log':
            # Log returns: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
            returns = np.diff(np.log(prices))
        elif method == 'simple':
            # Simple returns: (P_t - P_{t-1}) / P_{t-1}
            returns = np.diff(prices) / prices[:-1]
        else:
            raise ValueError(f"Unknown method: {method}. Use 'log' or 'simple'")

        # Create series with proper index
        returns_series = pd.Series(returns, index=self.data.index[1:])
        self.returns = returns_series

        if self.verbose:
            print(f"\n✓ Returns calculated ({method}):")
            print(f"  Number of returns: {len(returns)}")
            print(f"  Mean: {returns.mean():.8f}")
            print(f"  Std Dev: {returns.std():.8f}")
            print(f"  Min: {returns.min():.8f}")
            print(f"  Max: {returns.max():.8f}")
            print(f"  Skewness: {pd.Series(returns).skew():.4f}")
            print(f"  Kurtosis: {pd.Series(returns).kurtosis():.4f}")

        return returns_series

    def calculate_midpoint_returns(self):
        """
        Calculate returns using midpoint of bid-ask spread.

        Note: MT5 data doesn't include bid/ask, so this falls back to close prices.
        Use when bid/ask data is available from other sources.

        Returns:
        --------
        pd.Series
            Log returns calculated from midpoint prices
        """
        if 'bid' not in self.data.columns or 'ask' not in self.data.columns:
            if self.verbose:
                print("Bid/Ask data not available, using close prices instead")
            return self.calculate_returns()

        # Calculate midpoint
        midpoint = (self.data['bid'] + self.data['ask']) / 2
        self.prices = midpoint.values

        # Log returns
        returns = np.diff(np.log(midpoint.values))
        returns_series = pd.Series(returns, index=self.data.index[1:])
        self.returns = returns_series

        if self.verbose:
            print(f"\nMidpoint returns calculated:")
            print(f"Number of returns: {len(returns)}")
            print(f"Mean: {returns.mean():.8f}")
            print(f"Std Dev: {returns.std():.8f}")

        return returns_series

    def get_returns_array(self):
        """
        Get returns as numpy array for analysis.

        Returns:
        --------
        np.ndarray
            Array of returns
        """
        if self.returns is None:
            raise ValueError("No returns calculated. Call calculate_returns() first.")

        return self.returns.values

    def get_prices_array(self):
        """
        Get prices as numpy array.

        Returns:
        --------
        np.ndarray
            Array of prices
        """
        if self.prices is None:
            raise ValueError("No prices available. Call calculate_returns() first.")

        return self.prices

    def summary_statistics(self):
        """
        Print summary statistics of the loaded data.
        """
        if self.returns is None:
            raise ValueError("No returns calculated. Call calculate_returns() first.")

        print("\n" + "="*70)
        print("DATA SUMMARY STATISTICS")
        print("="*70)
        print(f"Symbol: {self.symbol}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Number of observations: {len(self.returns)}")
        print(f"Number of days: {(self.end_date - self.start_date).days}")

        if self.data is not None:
            print(f"\nPrice Statistics (Close):")
            print(f"  First: {self.data['close'].iloc[0]:.5f}")
            print(f"  Last: {self.data['close'].iloc[-1]:.5f}")
            print(f"  Min: {self.data['close'].min():.5f}")
            print(f"  Max: {self.data['close'].max():.5f}")
            print(f"  Mean: {self.data['close'].mean():.5f}")

        print(f"\nReturns Statistics:")
        print(f"  Mean: {self.returns.mean():.8f}")
        print(f"  Std Dev: {self.returns.std():.8f}")
        print(f"  Skewness: {self.returns.skew():.4f}")
        print(f"  Kurtosis: {self.returns.kurtosis():.4f}")
        print(f"  Min: {self.returns.min():.8f}")
        print(f"  Max: {self.returns.max():.8f}")

        # Check for data quality issues
        nan_count = self.returns.isna().sum()
        zero_count = (self.returns == 0).sum()
        zero_pct = 100 * zero_count / len(self.returns)

        print(f"\nData Quality:")
        print(f"  NaN values: {nan_count}")
        print(f"  Zero returns: {zero_count} ({zero_pct:.2f}%)")

        if nan_count > 0:
            print("  ⚠️  WARNING: Data contains NaN values!")
        if zero_pct > 5:
            print(f"  ⚠️  WARNING: High percentage of zero returns ({zero_pct:.1f}%)")

        print("="*70 + "\n")

    def __del__(self):
        """Cleanup: shutdown MT5 connection when object is destroyed."""
        self._shutdown_mt5()


def demo_usage_mt5():
    """Demonstrate MT5 data loading."""
    print("="*70)
    print("MMAR Data Loader - MetaTrader 5 Demo")
    print("="*70)

    # Initialize loader
    loader = DataLoader(
        symbol="EURUSD",
        start_date="2024-01-01",
        end_date="2024-01-31",  # Just 1 month for demo
        verbose=True
    )

    try:
        # Load data from MT5
        loader.load_from_mt5()

        # Calculate returns
        loader.calculate_returns(price_column='close', method='log')

        # Get summary
        loader.summary_statistics()

        # Get returns for analysis
        returns = loader.get_returns_array()
        print(f"Ready for analysis: {len(returns)} returns loaded")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("  1. MetaTrader 5 terminal is running")
        print("  2. You are logged into an account")
        print("  3. EURUSD symbol is available")
        print("  4. MetaTrader5 Python package is installed:")
        print("     pip install MetaTrader5")


def demo_usage_csv():
    """Demonstrate CSV data loading."""
    print("="*70)
    print("MMAR Data Loader - CSV Demo")
    print("="*70)

    loader = DataLoader(
        symbol="EURUSD",
        start_date="2024-01-01",
        end_date="2024-01-31",
        verbose=True
    )

    print("\nTo use CSV loading:")
    print("  loader.load_csv('path/to/eurusd_1min.csv')")
    print("  loader.calculate_returns()")
    print("  returns = loader.get_returns_array()")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# MMAR DATA LOADER - USAGE EXAMPLES")
    print("#"*70)

    # Try MT5 demo
    if MT5_AVAILABLE:
        demo_usage_mt5()
    else:
        print("\nMetaTrader5 not available. Showing CSV example instead.\n")
        demo_usage_csv()

    print("\n" + "#"*70)
    print("# RECOMMENDED USAGE")
    print("#"*70)
    print("""
from data_loader import DataLoader

# METHOD 1: Load from MT5 (Recommended)
loader = DataLoader(symbol="EURUSD",
                   start_date="2024-01-01",
                   end_date="2025-07-01")
loader.load_from_mt5()  # ← Automatically fetches from MT5
loader.calculate_returns()
returns = loader.get_returns_array()

# METHOD 2: Load from CSV (Fallback)
loader = DataLoader()
loader.load_csv("eurusd_data.csv")
loader.calculate_returns()
returns = loader.get_returns_array()
    """)
