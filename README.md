# maad-portfolio-optimizer
Polyglot Portfolio Optimizer: Mean Absolute Deviation (MAD) Linear Programming Engine

A high-performance, production-ready portfolio optimization and deployment pipeline that marries **Julia's computational efficiency** with **Python's robust data ecosystem**. This project implements a **Mean Absolute Deviation (MAD)** asset allocation framework solved via Linear Programming (LP), complete with an automated walk-forward backtester net of transaction friction and a live brokerage execution pipeline via the Alpaca API.

Designed to demonstrate rigorous quantitative research methodologies, production-grade software architecture, and end-to-end data pipeline engineering.

---

## 🚀 Architectural Overview

Modern Portfolio Theory (MPT) traditionally relies on Mean-Variance optimization, which requires solving a Quadratic Programming (QP) problem. This engine utilizes the **MAD framework**, reformulating the optimization problem into a **Linear Program (LP)**. This significantly improves computational scalability and allows the system to process massive datasets rapidly.

The project utilizes a **polyglot architecture**:
* **Performance Layer (Julia):** Handles high-dimensional linear programming constraints using `JuMP` and the `HiGHS` optimization solver.
* **Data & Analytics Layer (Python):** Orchestrates historical data ingestion (`yfinance`), rolling feature/return metrics calculations (`NumPy`, `Pandas`), performance evaluation, and visualization (`Matplotlib`).
* **Production Bridge (`JuliaCall` / `PythonCall`):** Facilitates zero-copy memory transfers and seamless cross-language execution between Python and Julia environments.

---

##  Key Features

### 1. Quantitative Research & Mathematical Optimization
* **LP Formulation:** Transforms non-linear absolute deviation objectives into a system of linear inequalities, solvable in polynomial time.
* **Walk-Forward Backtesting:** Simulates multi-year historical performance using a rolling lookback window ($252$ trading days) to eliminate lookahead bias.
* **Friction & Slippage Modeling:** Embeds a parameterized transaction cost penalty ($10\text{ bps}$ per unit of turnover) to mirror real-world market friction.
* **Advanced Risk Metrics:** Computes annualized Sharpe Ratios and exact Maximum Drawdown paths.

### 2. Software & Data Pipeline Engineering
* **Polyglot Integration:** Uses `juliacall` to compile and execute high-speed Julia optimization modules directly inside Python data routines.
* **Robust Exception Handling:** Implements defensive programming safeguards (e.g., graceful fallback to a cash allocation if optimization encountering singular matrices or infeasible solutions).
* **Live Automated Rebalancing:** A fully-integrated live trading script (`live_trader.jl`) that pulls live market data, re-optimizes weights, closes legacy positions, and submits fractional notional market orders via Alpaca REST clients.

---

##  Mathematical Model

The model minimizes the Mean Absolute Deviation of portfolio returns from their mean, subject to a minimum target return and diversification bounds:

$$\min_{w, y} \frac{1}{T} \sum_{t=1}^{T} y_t$$

**Subject to:**
* **Deviation Constraints:** $$y_t \ge \sum_{i=1}^{N} (R_{t,i} - \bar{R}_i) w_i \quad \forall t \in \{1, \dots, T\}$$
    $$y_t \ge -\sum_{i=1}^{N} (R_{t,i} - \bar{R}_i) w_i \quad \forall t \in \{1, \dots, T\}$$
* **Target Return:** $\sum_{i=1}^{N} \bar{R}_i w_i \ge \rho$ *(where $\rho$ is the daily target return)*
* **Fully Invested Budget:** $\sum_{i=1}^{N} w_i = 1.0$
* **Long-Only & Diversification Bounds:** $0 \le w_i \le w_{\max} \quad \forall i \in \{1, \dots, N\}$ *(e.g., maximum $30\%$ allocation per asset)*

---

## Repository Structure

* **`mad_solver.jl`**: Core Julia script housing the mathematical optimization function `run_mad_optimization`. Accepts historical return matrices from Python, sets up variables/objectives/constraints via `JuMP`, and returns optimal weight vectors.
* **`backtest.py`**: Python-driven framework that downloads asset historical data (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `JPM`, `GS`), executes the walk-forward simulation loop, factors in turnover penalties, and generates a comparative analytics dashboard against an equal-weight benchmark.
* **`live_trader.jl`**: Production script bridging live environments. Embeds Python code inside Julia to use Alpaca SDKs for asset pulling and order submittal while executing the native Julia optimization code locally.

---

## 📊 Performance & Verification

The framework produces a comprehensive performance output directly into standard streams alongside equity curves:
