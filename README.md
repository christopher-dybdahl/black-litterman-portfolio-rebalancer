# Black-Litterman Portfolio Rebalancer

**Portfolio Management System with Black-Litterman Optimization**

A comprehensive Python implementation of portfolio optimization and rebalancing using the Black-Litterman model, built with real financial data from Refinitiv Eikon. The system enables portfolio management strategies with dynamic rebalancing, market view integration, and performance analysis.

## Overview

This project implements an advanced portfolio management system that combines:
- **Black-Litterman Model**: Bayesian approach to portfolio optimization incorporating market views
- **Dynamic Rebalancing**: Automated portfolio rebalancing with configurable frequencies and thresholds
- **Market Triggers**: View activation based on market conditions (VIX, EUR/USD, yield curves)
- **Performance Analysis**: Comprehensive analytics including Sharpe ratios, drawdown analysis, and benchmarking

The system manages multi-asset portfolios across:
1. **Equity Assets**: Russell 3000 (.RUA), ACWI Ex-US (ACWX.O)
2. **Fixed Income**: Core Aggregate Bond (AGG)
3. **Alternative Assets**: Real Estate (.SPCBMICUSRE), Emerging Markets (.MERH0A0), Commodities (.BCOM)
4. **Benchmark**: S&P 500 (.SP500)

## Key Features

- **Eikon Data Integration**: Real-time and historical data from Refinitiv Eikon
- **Black-Litterman Optimization**: Incorporates investor views into mean-variance optimization
- **Trigger-Based Rebalancing**: Market condition-based view activation using VIX, currency, and yield indicators
- **Multi-Frequency Analysis**: Support for daily, weekly, monthly, and quarterly rebalancing
- **Transaction Cost Modeling**: Realistic cost assumptions for portfolio turnover
- **Comprehensive Analytics**: Performance metrics, risk analysis, and visualization tools

## Project Structure

```
├── main files
│   ├── DataHandler.py          # Eikon data extraction and preprocessing
│   ├── PortfolioRebalancer.py  # Core portfolio optimization and rebalancing logic
│   ├── PortfolioAnalyzer.py    # Performance analysis and metrics calculation
│   └── main.py                 # Main execution script and configuration
├── data files
│   ├── df_1.csv               # Primary asset price data
│   ├── df_2.csv               # Extended asset universe data
│   ├── df_bm.csv              # Benchmark (S&P 500) data
│   └── df_index.csv           # Market indicators (VIX, EUR/USD, yields)
├── outputs
│   └── plots/                 # Generated visualizations and analysis charts
├── project files
│   ├── pyproject.toml         # Poetry dependency management
│   ├── poetry.lock           # Locked dependencies
│   └── README.md             # This file
```

## Requirements

- **Python**: ^3.11
- **Core Libraries**: pandas ^2.3.3, matplotlib ^3.10.6, numpy
- **Data Source**: eikon ^1.1.18 (Refinitiv Eikon API)
- **Additional**: scipy, scikit-learn (for optimization utilities)

## Installation

1. **Clone the repository:**
   ```powershell
   git clone <repository-url>
   cd black-litterman-portfolio-rebalancer
   ```

2. **Install dependencies using Poetry:**
   ```powershell
   poetry install
   ```

   Or using pip:
   ```powershell
   pip install pandas matplotlib eikon numpy scipy scikit-learn
   ```

3. **Configure Eikon Access:**
   - Replace `'SECRET_KEY'` in `DataHandler.py` with your Eikon App Key
   - Ensure Eikon Desktop/Workspace is running for data access

## Usage

### Quick Start

1. **Data Collection:**
   ```python
   # Update DataHandler.py with your date range
   python DataHandler.py
   ```

2. **Run Portfolio Optimization:**
   ```python
   # Execute main portfolio analysis
   python main.py
   ```

### Configuration

All portfolio parameters are configured in `main.py`:

```python
# Portfolio Configuration
cash_percent = 0.05          # Cash allocation percentage
transaction_costs = 0.001    # Transaction cost assumption
initial_investment = 10000   # Starting portfolio value

# Rebalancing Parameters
freqs = [1, 7, 30, 90]      # Rebalancing frequencies (days)
thresholds = [(0.2, 0.3)]   # Weight deviation thresholds

# Asset Allocation Strategies
strategies = np.array([
    [0.2, 0.1, 0.7],        # Conservative: 20% equity, 10% intl, 70% bonds
    [0.5, 0.2, 0.3]         # Aggressive: 50% equity, 20% intl, 30% bonds
])

# Black-Litterman Views
p = np.array([[-1, 1, 0], [0.5, 0.5, -1], [-1, 0.7, 0.3]])  # View vectors
q = np.array([[0.05], [0.04], [0.1]])                        # Expected returns
omega = np.array([[0.2**2, 0, 0], [0, 0.2**2, 0], [0, 0, 0.2**2]])  # View uncertainty
```

### Market Triggers

Configure market condition triggers for dynamic view activation:

```python
# Trigger Conditions
trigger = [('.VIX', 'Rel'), ('EUR=', 'Abs'), ('.MRILT', 'Abs')]
trigger_thresholds = [(-100, 0.1), (0, 1.1), (-10, 0.3)]
```

## Black-Litterman Model

### Mathematical Framework

The Black-Litterman model combines market equilibrium with investor views:

- **Equilibrium Returns**: π = δΣw_eq
- **Posterior Mean**: μ = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹π + P'Ω⁻¹Q]
- **Posterior Covariance**: Σ_bar = Σ + [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹

### View Integration

```python
# Example: International outperforming domestic by 5%
P = [-1, 1, 0]    # Portfolio picking vector
Q = [0.05]        # Expected outperformance
Ω = [0.04]        # View confidence (variance)
```

## Performance Analytics

### Generated Visualizations

- **Portfolio Returns**: Cumulative returns across rebalancing frequencies
- **Weight Evolution**: Dynamic allocation changes over time  
- **Drawdown Analysis**: Maximum drawdown periods and recovery
- **Sharpe Ratio Comparison**: Risk-adjusted performance metrics
- **Benchmark Comparison**: Performance vs S&P 500

### Key Metrics

```python
# Sharpe Ratio Calculation
def sharpe(risk_free, portfolio_returns, n=255):
    return_mean = portfolio_returns.mean() * n
    return_sigma = portfolio_returns.std() * np.sqrt(n)
    return (return_mean - risk_free) / return_sigma
```

## Data Sources

### Primary Assets (df_1.csv)
- **.RUA**: Russell 3000 Index (US Broad Market)
- **ACWX.O**: iShares ACWI Ex-US ETF (International Equity)
- **AGG**: iShares Core US Aggregate Bond ETF

### Extended Universe (df_2.csv)
- **.SPCBMICUSRE**: S&P Real Estate Index
- **.MERH0A0**: MSCI Emerging Markets Index
- **.BCOM**: Bloomberg Commodity Index

### Market Indicators (df_index.csv)
- **.SPVXSPID**: S&P 500 VIX Short-Term Futures Index
- **EUR=**: EUR/USD Exchange Rate
- **.MRILT**: US 10-Year Treasury Yield

## Architecture

### Core Classes

```python
class PortfolioRebalancer:
    def rebalance_blacklitterman(p, q, omega, tau, delta)  # BL optimization
    def rebalance_bl_trigger(trigger, df_index, thresholds)  # Conditional activation
    def simulate_rebalancing(initial_investment)            # Portfolio simulation

class PortfolioAnalyzer:
    def sharpe(risk_free, portfolio_returns)               # Risk-adjusted returns
    def max_sharpe_ratio(freqs, risk_free)                 # Optimal frequency analysis

class DataHandler:
    def eikonimport(start_date, end_date)                  # Historical data extraction
```

## Technical Implementation

- **Bayesian Optimization**: Black-Litterman posterior distribution calculation
- **Matrix Operations**: Efficient linear algebra using NumPy
- **Time Series Analysis**: Pandas-based financial data manipulation
- **Visualization**: Matplotlib-based performance charts and analytics
- **API Integration**: Seamless Eikon data pipeline

## Research Applications

This implementation demonstrates:
- **Modern Portfolio Theory**: Mean-variance optimization with Bayesian updates
- **Behavioral Finance**: Integration of subjective market views
- **Risk Management**: Dynamic allocation based on market conditions
- **Quantitative Finance**: Systematic approach to portfolio management

## Performance Considerations

- **Memory Efficiency**: Chunked data processing for large time series
- **Computational Speed**: Vectorized operations for portfolio calculations
- **Data Quality**: Robust handling of missing values and market holidays
- **Scalability**: Modular design for additional assets and strategies

## Development

When extending functionality:

1. **New Assets**: Add instruments to data extraction in `DataHandler.py`
2. **New Views**: Modify P, Q, Ω matrices in main configuration
3. **New Triggers**: Extend trigger logic in `PortfolioRebalancer.py`
4. **New Metrics**: Add analytics to `PortfolioAnalyzer.py`

## Academic Context

This project demonstrates advanced concepts in:
- **Portfolio Optimization**: Black-Litterman model implementation
- **Financial Engineering**: Systematic portfolio management
- **Data Science**: Large-scale financial data analysis
- **Risk Management**: Dynamic allocation strategies

## License

MIT License
