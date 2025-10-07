import numpy as np
import pandas as pd

from PortfolioRebalancer import PortfolioRebalancer


class PortfolioAnalyzer:
    def __init__(self, rebalancer: PortfolioRebalancer):
        self.rebalancer = rebalancer
        self.freqs = None

    def sharpe(self, risk_free: float, portfolio_returns: pd.DataFrame, n: int = 255):
        portfolio_returns = portfolio_returns.iloc[1:]
        return_mean = portfolio_returns.mean() * n
        return_sigma = portfolio_returns.std() * np.sqrt(n)
        return (return_mean - risk_free) / return_sigma

    def max_sharpe_ratio(self, freqs: list, risk_free: float, plot_freqs: bool = True, plot_sharpe_ratio: bool = True):
        portfolios_returns = self.rebalancer.plot_frequencies(freqs, plot=plot_freqs,
                                                              plot_sharpe_ratio=plot_sharpe_ratio, risk_free=risk_free)

        high_sharpe_ratio = -np.inf
        high_portfolio = None
        sharpe_ratios = []

        for portfolio in portfolios_returns.columns:
            sharpe_ratio = self.sharpe(risk_free, portfolios_returns[portfolio])
            sharpe_ratios.append(sharpe_ratio)

            if sharpe_ratio > high_sharpe_ratio:
                high_sharpe_ratio = sharpe_ratio
                high_portfolio = portfolio

        print(f'Sharpe ratios: {sharpe_ratios}')

        return high_portfolio
