using ProgressBars, StaticArrays, Plots, Statistics

struct BlackScholesModel
    dt::Float64
    NTimeSteps::Int64
    NSituationsBatch::Int64
    #volatitility
    sigma::Float64
    #drift rate: analogous to underlying mean interest
    mu::Float64
    #interest rate of underyling, risk-free bond
    r::Float64
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

#since we are recording all timesteps, we can instead propegate only up to a certain time, without tracking.

function simulate_system_no_track(system::BlackScholesModel,initial_price::Float64)
    sigma::Float64 = system.sigma
    mu::Float64 = system.mu
    batch_output = zeros((system.NSituationsBatch, 2))

    @. batch_output[:,1] = initial_price

    N_simuls = system.NSituationsBatch
    dt::Float64 = system.dt
    #time integration part
    new_price = initial_price * ones(N_simuls)
    cdrift = (mu - 0.5 * sigma^2) * dt
    #alternatively, use log decomposition, along with cumsum - may speed things up.
    for i in 2:system.NTimeSteps
        noise = @. sqrt(dt) * randn(N_simuls)
        all_drift = @. exp(cdrift + sigma * noise)
        @. new_price = new_price * all_drift
        
    end
    batch_output[:,2] = new_price
    return batch_output
end

function batch_simulator_no_track(system::BlackScholesModel,initial_price::Float64,N_batches::Int64 = 20)
    N_simuls = system.NSituationsBatch
    
    
    all_output = Array{Float64}(undef,(N_simuls * N_batches, 2))
    all_output[:,1] .= initial_price


    #multithread - no race condition due to preallocation
    @Threads.threads for k in ProgressBar(0:N_batches-1)
        local_batch_output = simulate_system(system,initial_price)
        
        @. all_output[k*N_simuls+1:(k+1)*N_simuls,2] = local_batch_output[:,2]

    end
    return all_output
end


function evaluate_call_position_BS(K::Float64,system::BlackScholesModel,initial_price::Float64,N_batches::Int64 = 20,confidence_level::Float64 = 0.95)
    #simulate system - rest of algorithm can be used for any simulated result, not just BS.
    total_time = system.dt * system.NTimeSteps
    final_prices = batch_simulator_no_track(system,initial_price,N_batches)[:,2]

    #we represent the "backward" propegated interest on the risk-free bond to adjust the potential _gain_ of the call
    adjusted_payout = exp(-total_time * system.r) .* (final_prices .- K)

    

    #calculate CVAR using sorting and selecting worst case scenarios

    #realistically, we would bin the data as sorting cost can skyrocket, 
    #but this is more accurate.
    worst_to_best = sort(adjusted_payout)

    cut_index = ceil(Int64, (1-confidence_level) * length(worst_to_best))

    mean_worst_case_loss = mean(worst_to_best[1:cut_index+1])

    return worst_to_best, mean_worst_case_loss, cut_index
end

function evaluate_put_position_BS(K::Float64,system::BlackScholesModel,initial_price::Float64,N_batches::Int64 = 20,confidence_level::Float64 = 0.95)
    #simulate system
    total_time = system.dt * system.NTimeSteps
    final_prices = batch_simulator_no_track(system,initial_price,N_batches)[:,2]

    #we represent the "backward" propegated interest on the risk-free bond to adjust the potential _gain_ of the call
    adjusted_payout = exp(-total_time * system.r) .* (K .-final_prices)

    
    #calculate CVAR using sorting and selecting worst case scenarios

    worst_to_best = sort(adjusted_payout)

    cut_index = ceil(Int64, (1-confidence_level) * length(worst_to_best))

    mean_worst_case_loss = mean(worst_to_best[1:cut_index+1])

    return worst_to_best, mean_worst_case_loss, cut_index
end

#conditions reminiscent of the past two decade for SNP 500 - brute-guessed numbers
test = BlackScholesModel(0.01,1000,20000,0.14,0.04,0.025)

payouts, mwcloss, cut_index = evaluate_put_position_BS(1025.0,test,1000.0,20,0.95)

hist = histogram(payouts, color = "green", xlabel = "Payout", 
ylabel = "Simulated frequencies of payouts", title = "CVar visualization", 
label = "Simulated payoffs and losses", normalize = :pdf, dpi = 900)
vline!(hist,[payouts[cut_index+1]], ls = :dash, color = "red", label = "Confidence cut at 95%")
vline!(hist, [mwcloss], ls = :dash, color = "blue", label = "Mean loss beyond confidence cut")

t = LinRange(-30,70,1000)

o = std(payouts)
mu = mean(payouts)
y = @. 1/sqrt(2pi * o^2) * exp(-(t-mu)^2/(2o^2))
plot!(hist, t, y, label = "Fitted normal distribution", color = "orange", linewidth = 2.0)


#simulated_movements = batch_simulator(test,1000.0,20)
#number_to_display = 100
#colors = [RGB(0.1,1 - n/number_to_display,0.5*n/number_to_display) for n in 1:number_to_display]'
#times = [i*test.dt for i in 1:test.NTimeSteps]
#plot0 = plot(times, [simulated_movements[n,:] for n in 1:number_to_display], 
#title = "Stock Price under BS model", legend = false, 
#linewidth = 0.5, linecolors = colors, xlabel = "Time", ylabel = "Possible future price", dpi = 500)

