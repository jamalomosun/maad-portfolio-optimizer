# mad_solver.jl
using JuMP
using HiGHS
using Statistics

function run_mad_optimization(R_input)
    # Convert the incoming Python array into a native Julia Matrix
    R = Matrix{Float64}(R_input)
    
    T, N = size(R)
    mean_returns = mean(R, dims=1)[:]
    
    target_return = 0.0002      # 0.02% daily target (realistic constraint)
    max_weight_per_asset = 0.30 # Max 30% per asset
    
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    @variable(model, 0 <= w[1:N] <= max_weight_per_asset) 
    @variable(model, y[1:T] >= 0)                         
    
    @objective(model, Min, (1/T) * sum(y[t] for t in 1:T))
    #@constraint(model, sum(mean_returns[i] * w[i] for i in 1:N) >= target_return)
    @constraint(model, sum(w[i] for i in 1:N) == 1.0)
    
    for t in 1:T
        deviation_expr = sum((R[t, i] - mean_returns[i]) * w[i] for i in 1:N)
        @constraint(model, y[t] >= deviation_expr)
        @constraint(model, y[t] >= -deviation_expr)
    end
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return value.(w)
    else
        error("No feasible solution found.")
    end
end