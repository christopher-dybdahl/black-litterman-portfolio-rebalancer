import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PortfolioRebalancer:

    def __init__(self, strategy: np.ndarray, freq: int, cash_percent: float, interest: float, transaction_costs: float,
                 thresholds: list, data: pd.DataFrame, **kwargs):
        # Parameters
        self.__strategy = strategy
        self.__freq = freq
        self.__cash_percent = cash_percent  # Percentage of the portfolio to keep as cash
        self.__interest = interest  # Annual fixed interest rate for the cash position
        self.__transaction_costs = transaction_costs
        self.__thresholds = thresholds
        self.__kwargs = kwargs

        # Data
        self.__assets = data.columns
        self.__portfolio_values_list = None
        self.__portfolio_returns = None
        self.__portfolio_values = None
        self.__portfolio_cumulative_returns = None
        self.__historical_prices = data.dropna()
        self.__historical_returns = self.__historical_prices.pct_change().dropna()
        self.__historical_values = None
        self.__cumulative_returns = None

        # Benchmark
        if 'benchmark' in self.__kwargs:
            benchmark = self.__kwargs['benchmark']
            self.__benchmark = True
            self.__benchmark_prices = benchmark.dropna()
            self.__benchmark_returns = self.__benchmark_prices.pct_change().dropna()
        else:
            self.__benchmark = False
            self.__benchmark_prices = None
            self.__benchmark_returns = None

        # Cash
        self.__historical_cash = None
        self.__historical_cash_weights = None

        # Weights
        self.__initial_investment = 0
        self.__historical_weights = None

        # Black Litterman
        self.__BL = False
        self.__p = None
        self.__q = None
        self.__omega = None
        self.__tau = None
        self.__delta = None
        self.__BL_strategy = {}
        self.__eq_expected_returns = {}
        self.__BL_expected_returns = {}

        # Trigger
        self.__trigger_index = False
        self.__trig_type = None
        self.__trig_data = None
        self.__trig_thresholds = None

        # Status
        self.__rebalanced = False

    def sharpe(self, risk_free: float, n: int = 255, **kwargs):
        if 'portfolio_returns' in kwargs:
            portfolio_returns = kwargs['portfolio_returns']
        else:
            portfolio_returns = self.__portfolio_returns
        return_mean = portfolio_returns.mean() * n
        return_sigma = portfolio_returns.std() * np.sqrt(n)
        return (return_mean - risk_free) / return_sigma

    def rebalance_bl_trigger(self, trig_type: list, trig_data: pd.DataFrame, trig_thresholds: list):
        print('Index trigger activated')
        self.__trigger_index = True
        self.__trig_type = trig_type
        self.__trig_data = trig_data.dropna()
        self.__trig_thresholds = trig_thresholds

    def __check_trigger(self, end):
        closest_end = min(self.__trig_data.index, key=lambda x: (x - end))
        current_trig_experience = self.__trig_data.loc[:end]
        trig_activate = []
        for i in range(len(self.__trig_type)):
            index_name, trigger = self.__trig_type[i]

            if trigger == 'Constant':
                trig_activate.append(True)
            elif trigger == 'Off':
                trig_activate.append(False)
            elif trigger == "Abs":
                current_trig_abs = current_trig_experience[index_name].iloc[-1]
                lower_threshold, upper_threshold = self.__trig_thresholds[i]

                if current_trig_abs < lower_threshold or current_trig_abs > upper_threshold:
                    trig_activate.append(True)
                else:
                    trig_activate.append(False)
            elif trigger == "Rel":
                current_trig_rel = current_trig_experience[index_name].pct_change().iloc[-1]
                lower_threshold, upper_threshold = self.__trig_thresholds[i]

                if current_trig_rel < lower_threshold or current_trig_rel > upper_threshold:
                    trig_activate.append(True)
                else:
                    trig_activate.append(False)
            else:
                raise NameError(f'Trigger type for {index_name} not recognized')

        return trig_activate

    def rebalance_blacklitterman(self, p: np.ndarray, q: np.ndarray, omega: np.ndarray, tau: float, delta: float):
        print('Black Litterman activated')
        self.__BL = True
        self.__p = p
        self.__q = q
        self.__omega = omega
        self.__tau = tau
        self.__delta = delta

    def __blacklitterman(self, p, q, tau, delta, omega, eq_strategy, start, end):
        current_experience = self.__historical_returns.loc[start:end]  # Filter out future samples
        sigma = current_experience.cov()  # .to_numpy(dtype=float) # Calculate covariance matrix
        pi = delta * sigma @ eq_strategy.reshape((len(eq_strategy), 1))  # BL implied equilibrium returns
        # omega = np.diag(np.diag(tau * p @ sigma @ p.T)) # TODO: Issue with volatility calculation
        omega_inv = np.linalg.inv(omega)  # Precision
        m_bar_inv = np.linalg.inv(
            np.linalg.inv(tau * sigma) + p.T @ omega_inv @ p)  # Calculate the covariance matrix of view?
        mu = m_bar_inv @ (
                np.linalg.inv(tau * sigma) @ pi + p.T @ omega_inv @ q)  # Calculate the BL implied risk premium
        sigma_bar = (1 - tau) * sigma + m_bar_inv  # Calculate the posterior covariance matrix
        new_weights = np.linalg.inv(sigma_bar) @ mu / delta
        new_weights_out = pd.Series(new_weights[0].tolist(), index=self.__assets)

        self.__BL_strategy[end] = new_weights_out.tolist()
        self.__eq_expected_returns[end] = pi.T.iloc[0]
        self.__BL_expected_returns[end] = mu.T.iloc[0]

        return new_weights_out

    def __check_thresholds(self, current_total_value, current_weights, target_weights):
        target_values = current_total_value * np.array(target_weights)
        current_values = current_total_value * np.array(current_weights)
        for current, target, (lower_threshold, upper_threshold) in zip(current_values, target_values,
                                                                       self.__thresholds):
            deviation = np.divide(np.abs(current - target), target, out=np.zeros_like(np.abs(current - target)),
                                  where=target != 0)
            if deviation < lower_threshold or deviation > upper_threshold:
                return True
        return False

    def simulate_rebalancing(self, initial_investment: float):
        self.__rebalanced = True
        self.__portfolio_values_list = []
        self.__historical_weights = []
        self.__historical_cash = []
        self.__historical_cash_weights = []
        self.__initial_investment = initial_investment
        current_weights = np.array(self.__strategy)
        current_units = initial_investment * current_weights / self.__historical_prices.iloc[0]  # Units of each asset
        current_cash = self.__cash_percent * initial_investment

        self.__portfolio_values_list.append(self.__initial_investment)
        self.__historical_weights.append(current_weights * (1 - self.__cash_percent))
        self.__historical_cash.append(current_cash)
        self.__historical_cash_weights.append(self.__cash_percent)

        for index, prices in enumerate(self.__historical_prices.iloc[:-1].itertuples(index=False), 1):

            date = self.__historical_prices.index[index]
            current_prices = np.array(prices)
            current_values = current_units * current_prices
            current_weights = current_values / np.sum(current_values)
            current_value_ex_cash = np.sum(current_values) - current_cash
            current_cash *= 1 + self.__interest
            current_total_value = current_value_ex_cash + current_cash
            current_cash_percent = current_cash / current_total_value

            if index % self.__freq == 0 and self.__check_thresholds(current_total_value, current_weights,
                                                                    self.__strategy):
                if self.__BL and index > 30:
                    if self.__trigger_index:
                        triggers = self.__check_trigger(date)
                        p_in_use = np.array([p if trigger else p * 0 for p, trigger in zip(self.__p, triggers)])
                    else:
                        p_in_use = self.__p
                    current_strategy = self.__blacklitterman(p_in_use, self.__q, self.__tau, self.__delta, self.__omega,
                                                             self.__strategy, self.__historical_prices.index[0], date)
                else:
                    current_strategy = self.__strategy

                current_weights, total_transaction_costs = self.__rebalance_portfolio(current_total_value,
                                                                                      current_weights, current_strategy)
                current_units = (current_total_value - total_transaction_costs) * current_weights / current_prices
                current_values = current_units * current_prices
                current_value_ex_cash = np.sum(current_values) - current_cash
                current_total_value = current_value_ex_cash + current_cash
                current_cash_percent = self.__cash_percent

            self.__portfolio_values_list.append(current_total_value)
            self.__historical_weights.append(current_weights * (1 - current_cash_percent))
            self.__historical_cash.append(current_cash)
            self.__historical_cash_weights.append(current_cash_percent)

        self.__calculate_returns()
        self.__calculate_values()

        return self.__portfolio_returns

    def __rebalance_portfolio(self, current_value: np.ndarray, current_weights: np.ndarray, target_weights: np.ndarray):
        current_values = current_value * current_weights
        desired_values = current_value * target_weights
        dollar_transactions = desired_values - current_values
        transaction_costs = np.abs(dollar_transactions) * self.__transaction_costs
        total_transaction_costs = np.sum(transaction_costs)
        adjusted_current_value = current_value - total_transaction_costs
        adjusted_values = current_values + dollar_transactions - transaction_costs
        new_weights = adjusted_values / adjusted_current_value

        return new_weights, total_transaction_costs

    def plot_weights(self, date: pd.Timestamp):
        if not self.__rebalanced:
            raise UserWarning('Simulate rebalancing before plotting weights')
        elif not self.__BL:
            raise UserWarning('Initiate Black Litterman before plotting weights')

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))  # Set the size of the figure

        bl_date = min(self.__BL_strategy.keys(), key=lambda x: abs(x - date))

        # Plot each set of weights as a bar chart
        ax.bar(self.__assets, self.__strategy, label='Equilibrium Strategy', width=0.4, align='center', color='blue')
        ax.bar(self.__assets, self.__BL_strategy[bl_date], label='BL Strategy', width=0.4, align='edge', color='red')

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax.set_ylabel('Weights')
        ax.set_title(f'Optimal Portfolio Weights: {bl_date}')
        ax.set_xticks(self.__assets)
        ax.set_xticklabels(self.__assets)
        ax.legend()

        # Set the y-axis formatter to percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.show()

    def plot_strat_returns(self, date: pd.Timestamp):
        if not self.__rebalanced:
            raise UserWarning('Simulate rebalancing before plotting expected returns')
        elif not self.__BL:
            raise UserWarning('Initiate Black Litterman before plotting expected returns')

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))  # Set the size of the figure

        bl_date = min(self.__BL_strategy.keys(), key=lambda x: abs(x - date))

        # Plot each set of returns as a bar chart
        ax.bar(self.__assets, self.__eq_expected_returns[bl_date], label='Equilibrium Returns', width=0.4,
               align='center',
               color='blue')
        ax.bar(self.__assets, self.__BL_expected_returns[bl_date], label='BL Returns', width=0.4, align='edge',
               color='red')

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax.set_ylabel('Returns')
        ax.set_title(f'Expected returns: {bl_date}')
        ax.set_xticks(self.__assets)
        ax.set_xticklabels(self.__assets)
        ax.legend()

        # Set the y-axis formatter to percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.show()

    def plot_historical_weights(self):
        if not self.__rebalanced:
            raise UserWarning('Simulate rebalancing before plotting weights')

        # Convert the list of weights to a DataFrame
        weights_df = pd.DataFrame(self.__historical_weights, index=self.__historical_prices.index,
                                  columns=self.__assets)
        weights_df['Cash'] = self.__historical_cash_weights

        # Plot settings
        plt.figure(figsize=(12, 6))
        plt.title('Historical Weight Allocations in the Portfolio')

        # Formatting the x-axis to show dates clearly
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Plotting
        plt.stackplot(weights_df.index, weights_df.T, labels=weights_df.columns)

        # Labels and legend
        plt.xlabel('Date')
        plt.ylabel('Weight Allocation')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def __calculate_returns(self):
        self.__cumulative_returns = (1 + self.__historical_returns).cumprod().dropna() - 1

        if self.__rebalanced:
            self.__portfolio_returns = pd.Series(self.__portfolio_values_list,
                                                 index=self.__historical_prices.index).pct_change().rename('Portfolio')
            self.__portfolio_cumulative_returns = (1 + self.__portfolio_returns).cumprod().dropna() - 1

        if self.__benchmark:
            self.benchmark_cumulative_returns = (1 + self.__benchmark_returns.loc[
                                                     self.__cumulative_returns.index[0]:]).cumprod().dropna() - 1

    def __calculate_values(self):
        # Calculate the absolute values for each asset over time
        self.__historical_values = (self.__historical_prices / self.__historical_prices.iloc[
            0]) * self.__initial_investment

        if self.__rebalanced:  # Dataframe for the rebalanced portfolio
            self.__portfolio_values = pd.Series(self.__portfolio_values_list,
                                                index=self.__historical_prices.index).rename('Portfolio')

        if self.__benchmark:
            self.benchmark_values = (self.__benchmark_prices.loc[self.__historical_values.index[0]:] /
                                     self.__benchmark_prices.loc[
                                         self.__historical_values.index[0]]) * self.__initial_investment

    def plot_returns(self):
        # Plot settings
        plt.figure(figsize=(12, 6))
        plt.title(f'Asset and Portfolio Returns Over Time (freq: {self.__freq}, strategy: {self.__strategy})')

        # Plot each asset
        for column in self.__cumulative_returns.columns:
            plt.plot(self.__cumulative_returns.index, self.__cumulative_returns[column], label=column,
                     linestyle='--')

        if self.__rebalanced:
            plt.plot(self.__portfolio_cumulative_returns.index, self.__portfolio_cumulative_returns, linewidth=2,
                     label='Portfolio Returns',
                     linestyle='solid')

        if self.__benchmark:
            plt.plot(self.benchmark_cumulative_returns.index, self.benchmark_cumulative_returns,
                     label='Benchmark', linewidth=2,
                     linestyle='--')

        # Formatting the x-axis to show dates clearly
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Labels and legend
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def plot_values(self):
        # Plot settings
        plt.figure(figsize=(12, 6))
        plt.title(f'Asset and Portfolio Value Over Time (freq: {self.__freq}, strategy: {self.__strategy})')

        # Plot each asset
        for column in self.__historical_values.columns:
            plt.plot(self.__historical_values.index, self.__historical_values[column], label=column, linestyle='--')

        if self.__rebalanced:
            plt.plot(self.__portfolio_values.index, self.__portfolio_values, linewidth=2,
                     label='Portfolio Value', linestyle='--')

        if self.__benchmark:
            plt.plot(self.benchmark_values.index, self.benchmark_values, label=self.benchmark_values.columns[0],
                     linewidth=2,
                     linestyle='--')

        # Formatting the x-axis to show dates clearly
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Labels and legend
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    def hwm(self, plot_hwm: bool = True):
        high_water_mark = self.__portfolio_cumulative_returns.cummax()

        if plot_hwm:
            # Plot settings
            plt.figure(figsize=(12, 6))
            plt.title(f'High Water Mark (strategy: {self.__strategy})')

            plt.plot(high_water_mark.index, high_water_mark, label=high_water_mark.name, linestyle='solid')

            # Formatting the x-axis to show dates clearly
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Labels and legend
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)
            plt.show()

        return high_water_mark

    def drawdown(self, plot_hwm: bool = True):
        hwm = self.__portfolio_cumulative_returns.cummax()
        drawdown = hwm - self.__portfolio_cumulative_returns
        drawdown = drawdown.div(hwm).replace(np.inf, np.nan).fillna(0)

        if plot_hwm:
            # Plot settings
            plt.figure(figsize=(12, 6))
            plt.title(f'Drawdown (strategy: {self.__strategy})')

            plt.plot(drawdown.index, drawdown, label=drawdown.name, linestyle='--')

            # Formatting the x-axis to show dates clearly
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Labels and legend
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)
            plt.show()

        return drawdown

    def plot_frequencies(self, freqs: list, initial_investment: float = 10000, plot: bool = False,
                         plot_sharpe_ratio: bool = False, risk_free: float = 0, plot_hwm: bool = False,
                         plot_drawdown: bool = False):
        portfolios_returns = pd.DataFrame(index=self.__historical_prices.index)
        portfolios_cumulative_returns = pd.DataFrame(index=self.__historical_prices.index)
        portfolios_high_water_mark = pd.DataFrame(index=self.__historical_prices.index)
        portfolios_drawdown = pd.DataFrame(index=self.__historical_prices.index)

        old_freq = self.__freq

        for freq in freqs:
            self.__freq = freq
            portfolios_returns[freq] = self.simulate_rebalancing(initial_investment)

            portfolios_cumulative_returns[freq] = (1 + portfolios_returns[freq]).cumprod().dropna() - 1

            portfolios_high_water_mark[freq] = portfolios_cumulative_returns[freq].cummax()

            drawdown = portfolios_cumulative_returns[freq].cummax() - portfolios_cumulative_returns[freq]
            drawdown[drawdown < 0] = 0
            portfolios_drawdown[freq] = drawdown

        self.__freq = old_freq

        if plot:
            plt.figure(figsize=(12, 6))
            plt.title(f'Portfolio Returns Over Time - Different Frequencies (strategy: {self.__strategy})')

            for portfolio in portfolios_cumulative_returns.columns:
                plt.plot(portfolios_cumulative_returns.index, portfolios_cumulative_returns[portfolio],
                         label=f'Frequency {portfolio}', linestyle='--')

            # Formatting the x-axis to show dates clearly
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Labels and legend
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)

            plt.show()

        if plot_sharpe_ratio:
            if not risk_free:
                risk_free = 0.01
                print(f'Assume risk free: {risk_free}')

            fig, ax = plt.subplots(figsize=(10, 6))  # Set the size of the figure
            ax.set_title(f'Sharpe Ratios For Different Frequencies')

            for portfolio in portfolios_cumulative_returns:
                sharpe_ratio = self.sharpe(risk_free, portfolio_returns=portfolios_returns[portfolio].iloc[1:])
                ax.bar(f'{portfolio}', sharpe_ratio, label=f'Frequency {portfolio}', width=0.4, align='center')

            # Add some text for labels, title, and custom x-axis tick labels, etc.
            ax.set_ylabel('Sharpe Ratio')
            ax.legend()

            plt.show()

        if plot_hwm:
            plt.figure(figsize=(12, 6))
            plt.title(f'High Water Mark - Different Frequencies (strategy: {self.__strategy})')

            for portfolio in portfolios_high_water_mark.columns:
                plt.plot(portfolios_high_water_mark.index, portfolios_high_water_mark[portfolio],
                         label=f'Frequency {portfolio}',
                         linestyle='--')

            # Formatting the x-axis to show dates clearly
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Labels and legend
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)

        if plot_drawdown:
            plt.figure(figsize=(12, 6))
            plt.title(f'Drawdown - Different Frequencies (strategy: {self.__strategy})')

            for portfolio in portfolios_drawdown.columns:
                plt.plot(portfolios_drawdown.index, portfolios_drawdown[portfolio], label=f'Frequency {portfolio}',
                         linestyle='--')

            # Formatting the x-axis to show dates clearly
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

            # Labels and legend
            plt.xlabel('Date')
            plt.ylabel('Drawdown')
            plt.legend()
            plt.grid(True)

        return portfolios_returns.dropna()
