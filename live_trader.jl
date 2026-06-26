# live_trader.jl
using PythonCall
using JuMP
using HiGHS
using Statistics

println("--- Initializing Live Brokerage Pipeline ---")

pyexec("""
import numpy as np
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime, timedelta

API_KEY = "YOUR_API_KEY"
SECRET_KEY = "YOUR_SECRET_KEY"

trade_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def fetch_live_market_data(tickers):
    print("Fetching live market data from Alpaca...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )
    
    bars = data_client.get_stock_bars(request_params).df
    prices = bars.reset_index().pivot(index='timestamp', columns='symbol', values='close')
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    return log_returns.to_numpy(), list(log_returns.columns)

def execute_rebalance(optimal_weights_dict):
    print("Executing trades via Alpaca API...")
    account = trade_client.get_account()
    portfolio_value = float(account.portfolio_value)
    
    trade_client.close_all_positions(cancel_orders=True)
    print("Closed existing positions. Allocating new capital...")
    
    for symbol, weight in optimal_weights_dict.items():
        if weight > 0.01: 
            target_dollar_amount = portfolio_value * weight
            market_order = MarketOrderRequest(
                symbol=symbol,
                notional=target_dollar_amount,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            trade_client.submit_order(order_data=market_order)
            print(f"Submitted BUY order for {symbol}: {target_dollar_amount:.2f}")
    
    print("Portfolio Rebalance Complete.")
""")

fetch_live_market_data = pyglobals["fetch_live_market_data"]
execute_rebalance = pyglobals["execute_rebalance"]

portfolio_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS"]

# Fetch Data
py_result = fetch_live_market_data(portfolio_tickers)
R = pyconvert(Matrix{Float64}, py_result[0])
assets = pyconvert(Vector{String}, py_result[1])

T, N = size(R)
mean_returns = mean(R, dims=1)[:]

# Run Optimizer
target_return = 0.001       
max_weight_per_asset = 0.30 

model = Model(HiGHS.Optimizer)
set_silent(model)

@variable(model, 0 <= w[1:N] <= max_weight_per_asset) 
@variable(model, y[1:T] >= 0)                         

@objective(model, Min, (1/T) * sum(y[t] for t in 1:T))
@constraint(model, sum(mean_returns[i] * w[i] for i in 1:N) >= target_return)
@constraint(model, sum(w[i] for i in 1:N) == 1.0)

for t in 1:T
    deviation_expr = sum((R[t, i] - mean_returns[i]) * w[i] for i in 1:N)
    @constraint(model, y[t] >= deviation_expr)
    @constraint(model, y[t] >= -deviation_expr)
end

println("Executing MAD Optimization Model...")
optimize!(model)

# Execute Trades
if termination_status(model) == MOI.OPTIMAL
    println("\n--- Optimal Portfolio Found ---")
    optimal_weights = value.(w)
    
    weight_dict = Dict{String, Float64}()
    for i in 1:N
        weight_dict[assets[i]] = optimal_weights[i]
        println("$(assets[i]): $(round(optimal_weights[i] * 100, digits=2))%")
    end
    
    println("\nSending execution instructions to brokerage...")
    execute_rebalance(weight_dict)
else
    println("No feasible solution found. Adjust constraints. No trades executed.")
end