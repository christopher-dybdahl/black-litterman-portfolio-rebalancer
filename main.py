import eikon as ek
import numpy as np
import pandas as pd

from PortfolioRebalancer import PortfolioRebalancer
from PortfolioAnalyzer import PortfolioAnalyzer

if __name__ == '__main__':
    df_1 = pd.read_csv('df_1.csv', index_col=0, header=0, parse_dates=True).astype(float).dropna()
    df_2 = pd.read_csv('df_2.csv', index_col=0, header=0, parse_dates=True).astype(float).dropna()
    df_bm = pd.read_csv('df_bm.csv', index_col=0, header=0, parse_dates=True).astype(float).dropna()
    df_index = pd.read_csv('df_index.csv', index_col=0, header=0, parse_dates=True).astype(float).dropna()

    start = max([df_1.index[0], df_2.index[0], df_bm.index[0], df_index.index[0]])
    df_1 = df_1.loc[start:]
    df_2 = df_2.loc[start:]
    df_bm = df_bm.loc[start:]
    df_index = df_index.loc[start:]

    cash_percent = 0.05
    interest = 0
    risk_free = 0.025
    transaction_costs = 0.001
    initial_investment = 10000

    freqs = [1, 7, 30, 90]
    freq = 1

    thresholds = [(0.2, 0.3), (0.2, 0.3), (0.2, 0.3)]
    strategies = np.array([[0.2, 0.1, 0.7], [0.5, 0.2, 0.3]])

    # Black Litterman
    p = np.array([[-1, 1, 0], [0.5, 0.5, -1], [-1, 0.7, 0.3]])  # Vector for the single view
    q = np.array([[0.05], [0.04], [0.1]])  # Scalar for the single view
    omega = np.array([[0.2 ** 2, 0, 0], [0, 0.2 ** 2, 0], [0, 0, 0.2 ** 2]])  # Variance for the view

    tau = 0.02  # Scaling factor (τ)
    delta = 2.5  # Risk

    # Trigger
    trigger = [('.VIX', 'Rel'), ('EUR=', 'Abs'), ('.MRILT', 'Abs')]
    trigger_thresholds = [(-100, 0.1), (0, 1.1), (-10, 0.3)]

    for strategy in strategies:
        rebalancer = PortfolioRebalancer(strategy, freq, cash_percent, interest, transaction_costs, thresholds, df_1,
                                         benchmark=df_bm)  # Initializing portfolio
        rebalancer.rebalance_blacklitterman(p, q, omega, tau, delta)  # Comment out if you do not want BL
        rebalancer.rebalance_bl_trigger(trigger, df_index,
                                        trigger_thresholds)  # Comment out if you do not want to trigger BL, that is always view active

        portfolio_cumulative_return = rebalancer.simulate_rebalancing(initial_investment)  # Rebalance for one frequency
        rebalancer.plot_returns()  # Plot cumulative returns
        rebalancer.plot_historical_weights()  # Plot weights
        rebalancer.plot_values()  # Plot absolute value of assets

        rebalancer.sharpe(risk_free)  # Plot sharp ratio
        rebalancer.hwm()  # Plot high watermark
        rebalancer.drawdown()  # Plot drawdown

        rebalancer.plot_frequencies(freqs, initial_investment, plot=True, plot_hwm=True,
                                    plot_drawdown=True)  # Plotting metrics for different frequencies

        rebalancer_analyzer = PortfolioAnalyzer(rebalancer)  # Initializing analyzer for maximizing sharpe ratio
        print(rebalancer_analyzer.max_sharpe_ratio(freqs, risk_free,
                                                   plot_freqs=False))  # Plotting sharpe ratio, and printing out best frequency

        # rebalancer.plot_weights(pd.to_datetime('2022-11-01')) # Plot weights before and after BL at a certain date
        # rebalancer.plot_strat_returns(pd.to_datetime('2022-11-01')) # Plot expected returns before and after BL at a certain date