using ProgressBars, StaticArrays, Plots, Statistics

struct BlackScholesModel
    dt::Float64
    NTimeSteps::Int64
    NSituationsBatch::Int64
    #volatitility
    sigma::Float64
    #drift rate: analogous to underlying mean interest
    mu::Float64
end

function simulate_system(system::BlackScholesModel,initial_price::Float64)
    sigma::Float64 = system.sigma
    mu::Float64 = system.mu
    batch_output = zeros((system.NSituationsBatch, system.NTimeSteps))

    @. batch_output[:,1] = initial_price

    N_simuls = system.NSituationsBatch
    dt::Float64 = system.dt
    #time integration part
    new_price = initial_price * ones(N_simuls)
    for i in 2:system.NTimeSteps
        noise = @. sqrt(dt) * randn(N_simuls)
        all_drift = @. exp((mu - 0.5 * sigma^2) * dt + sigma * noise)
        @. new_price = new_price * all_drift
        @inbounds batch_output[:,i] = new_price
    end
    return batch_output
end

function batch_simulator(system::BlackScholesModel,initial_price::Float64,N_batches::Int64 = 20)
    N_simuls = system.NSituationsBatch
    NT = system.NTimeSteps
    
    all_output = Array{Float64}(undef,(N_simuls * N_batches, NT))

    #multithread - no race condition due to preallocation
    @Threads.threads for k in ProgressBar(0:N_batches-1)
        local_batch_output = simulate_system(system,initial_price)
        @inbounds @. all_output[k*N_simuls+1:(k+1)*N_simuls,:] = local_batch_output

    end
    return all_output
end

#since we are recording all timesteps, we can do something like this

test = BlackScholesModel(0.01,1000,1000,0.001,0.001)
simulated_movements = batch_simulator(test,1000.0,20)


number_to_display = 100

colors = [RGB(0.1,1 - n/number_to_display,0.5*n/number_to_display) for n in 1:number_to_display]'
times = [i*test.dt for i in 1:test.NTimeSteps]

plot0 = plot(times, [simulated_movements[n,:] for n in 1:number_to_display], 
title = "Stock Price under BS model", legend = false, 
linewidth = 0.5, linecolors = colors, xlabel = "Time", ylabel = "Possible future price", dpi = 500)

