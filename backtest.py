# backtest.py
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt 
from juliacall import Main as jl

# Load the core Julia math engine
jl.include("mad_solver.jl") 
mad_optimize = jl.run_mad_optimization 

def fetch_historical_data(tickers, start, end):
    print("Fetching historical dataset...")
    prices = yf.download(tickers, start=start, end=end)['Close']
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns

def backtest_strategy_with_costs(returns_df, lookback_window=252, rebalance_freq=1, cost_bps=10):
    print(f"Starting Walk-Forward Backtest (Friction: {cost_bps} bps)...")
    
    n_days = len(returns_df)
    assets = returns_df.columns.tolist()
    cost_factor = cost_bps / 10000.0  
    
    strategy_returns = []
    previous_weights = np.zeros(len(assets))
    
    for t in range(lookback_window, n_days, rebalance_freq):
        historical_window = returns_df.iloc[t - lookback_window : t]
        R_matrix = historical_window.to_numpy() 
        
        try:
            # Pass matrix to Julia for optimization
            optimal_weights = mad_optimize(R_matrix) 
            optimal_weights = np.array(optimal_weights) # Ensure it is a numpy array
        except Exception as e:
            # Print the error so we know it failed, but hold cash to be safe
            print(f"Day {t}: Optimization failed. Going to cash. Error: {e}")
            optimal_weights = np.zeros(len(assets))
        
        # Calculate Turnover, Costs, and Net Return
        turnover = np.sum(np.abs(optimal_weights - previous_weights))
        transaction_costs = turnover * cost_factor
        
        next_day_returns = returns_df.iloc[t].to_numpy()
        gross_return = np.dot(optimal_weights, next_day_returns)
        net_return = gross_return - transaction_costs
        
        strategy_returns.append(net_return)
        previous_weights = optimal_weights
        
    return pd.Series(strategy_returns, index=returns_df.index[lookback_window::rebalance_freq])

def calculate_performance_metrics(returns_series, risk_free_rate=0.0):
    """Calculates Annualized Sharpe Ratio and Maximum Drawdown."""
    # 1. Sharpe Ratio
    daily_mean = returns_series.mean()
    daily_volatility = returns_series.std()
    
    # Avoid division by zero if volatility is flat
    if daily_volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = ((daily_mean - risk_free_rate) / daily_volatility) * np.sqrt(252)
    
    # 2. Maximum Drawdown
    cumulative_returns = (1 + returns_series).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return sharpe_ratio, max_drawdown

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS"]
    data = fetch_historical_data(tickers, "2020-01-01", "2024-01-01")
    
    # 1. Run Strategy
    portfolio_returns = backtest_strategy_with_costs(data, cost_bps=10)
    strategy_curve = (1 + portfolio_returns).cumprod()
    
    # 2. Calculate Benchmark (Equal-Weight)
    # We slice the data to start exactly when the strategy starts trading (after the lookback window)
    lookback_window = 252
    benchmark_returns = data.iloc[lookback_window:].mean(axis=1)
    benchmark_curve = (1 + benchmark_returns).cumprod()
    
    # ... [Keep plotting and initial strategy/benchmark calculations exactly as they are] ...

    # Calculate Risk Metrics
    mad_sharpe, mad_mdd = calculate_performance_metrics(portfolio_returns)
    bench_sharpe, bench_mdd = calculate_performance_metrics(benchmark_returns)
    
    # Print Final Stats
    print("\n" + "="*45)
    print("      QUANTITATIVE PERFORMANCE REPORT      ")
    print("="*45)
    print(f"{'Metric':<20} | {'MAD Strategy':<10} | {'Benchmark':<10}")
    print("-" * 45)
    print(f"{'Final Value':<20} | ${strategy_curve.iloc[-1]:<9.2f} | ${benchmark_curve.iloc[-1]:<9.2f}")
    print(f"{'Sharpe Ratio':<20} | {mad_sharpe:<10.2f} | {bench_sharpe:<10.2f}")
    print(f"{'Max Drawdown':<20} | {mad_mdd*100:<9.2f}% | {bench_mdd*100:<9.2f}%")
    print("="*45)
    
    # 3. Plot the Equity Curves
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_curve, label="MAD Strategy (Net of Costs)", color="#2ca02c", linewidth=2)
    plt.plot(benchmark_curve, label="Equal-Weight Benchmark", color="#1f77b4", linestyle="--", linewidth=2)
    
    # Formatting the Chart
    plt.title("MAD Optimization vs. Equal-Weight Benchmark (2021-2024)", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Portfolio Value ($)", fontsize=12)
    plt.legend(loc="upper left", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Display the Chart
    plt.tight_layout()
    plt.show()