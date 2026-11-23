# Black–Litterman Portfolio Rebalancing under Transaction Costs

When should a multi-asset portfolio rebalance: on the calendar, or when
weights drift past a threshold? And how should market views enter the answer?
This project compares rebalancing policies for a Black–Litterman portfolio
under realistic transaction costs.

Group final project for **MIT 15.433 Financial Markets**
("Evaluating Optimal Rebalancing Strategies for Diverse Asset Class
Portfolios").

## Problem

A diversified portfolio of US equity (Russell 3000), international equity
(ACWI ex-US), core fixed income (AGG), plus real estate, credit and
commodities, benchmarked against the S&P 500, drifts away from its target
allocation as prices move. Rebalancing restores the allocation but costs
money. The trade-off is evaluated along three axes:

1. **Calendar policies**: rebalance every 1, 7, 30 or 90 trading days.
2. **Threshold policies**: rebalance only when a weight breaches a band
   around its target (absolute and relative bands).
3. **View-driven rebalancing**: Black–Litterman views (`P`, `Q`, `Ω`,
   `τ = 0.02`) are activated by market triggers: VIX spikes, EUR/USD moves,
   and yield-curve shifts update the posterior expected returns, which
   shifts the target allocation itself.

Assumptions: 5% cash buffer, 10 bp proportional transaction costs,
Sharpe-ratio and drawdown evaluation against buy-and-hold and the benchmark.
Data from Refinitiv Eikon, 2016–2023.

## Black–Litterman machinery

Equilibrium returns are reverse-optimised from market-cap weights
(`Π = δ Σ w_mkt`) and blended with views via the standard posterior

```
E[R] = [(τΣ)⁻¹ + Pᵀ Ω⁻¹ P]⁻¹ [(τΣ)⁻¹ Π + Pᵀ Ω⁻¹ Q]
```

before mean–variance optimisation. Because views switch on and off with
market triggers, the target portfolio is time-varying: rebalancing policy
and view policy interact.

## What's here

```
DataHandler.py          # Eikon data extraction and preprocessing
PortfolioRebalancer.py  # BL posterior, triggers, rebalancing engine
PortfolioAnalyzer.py    # Sharpe, drawdown, high-water-mark analytics
main.py                 # experiment configuration and runs
plots/                  # results figures
```

Key figures in `plots/`: cumulative returns by rebalancing frequency
(`freq_returns.png`), drawdowns (`freq_drawdown.png`, `drawdown.png`),
weight evolution (`weights_1.png`, `weights_2.png`) and Sharpe comparison
(`sr.png`).

## Findings

- Gross of costs, calendar frequency barely moves the cumulative return
  path (`freq_returns.png`), but net of 10 bp costs, frequent rebalancing
  mainly means paying more often for nearly the same path.
- Threshold-band policies achieve comparable tracking with far fewer trades.
- Trigger-activated views concentrate turnover in regime shifts (notably
  the 2020 drawdown), which is where re-optimising the target allocation
  adds the most value.

## Setup

```bash
poetry install
poetry run python main.py
```

Requires a Refinitiv Eikon API key for fresh data; cached CSVs
(`df_1.csv`, `df_2.csv`, `df_bm.csv`, `df_index.csv`) reproduce the study
without one.
