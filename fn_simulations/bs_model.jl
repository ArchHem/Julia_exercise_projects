using ProgressBars, StaticArrays, Plots, Statistics, CSV, DataFrames, Statistics, Dates, Optim, LaTeXStrings

OTP_DATA_DF = CSV.read("fn_simulations/fn_data/OTP.csv", delim=",", DataFrame)
#OTP_DATA_DF = CSV.read("fn_simulations/fn_data/OTP_weekly.csv", delim=",", DataFrame)

function date_to_seconds(date::Date)

    #We do not know the eaxt closing time but it seldom matters (we are only interested in differences in time to accounts for uneven time-series data
    
    date = DateTime(date)
    seconds_passed = datetime2unix(date)
    return seconds_passed
end

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

function log_likelyhood(logratio::Vector{Float64},delta_t::Vector{Float64},
    mu::Float64,o::Float64)

    #uses the fact that the log of the rato of S_i+1 /S_i is normally distributed with spread proportial to sqrt(delta t)

    to_sum = @. -0.5 * (logratio - (mu - o^2 /2) * delta_t)^2 / (o^2 * delta_t) - log(sqrt(2pi * o^2 * delta_t))

    LLH = -sum(to_sum)

    return LLH

end

function estimate_BS_params(data::DataFrame)
    #assume CSV generated query from Yahoo Finance
    year_length = 3600 * 24 * 365

    dates_in_year = date_to_seconds.(data[!,"Date"]) / year_length
    delta_times_in_year = dates_in_year[2:end]-dates_in_year[1:end-1]
    prices = Array(data[!,"Close"]) 
    
    valid_logdiff = log.(prices[2:end]) .- log.(prices[1:end-1])
    
    #see:
    #https://www.wolframalpha.com/input
    #?i=d%2Fdu+ln%28exp%28-%28x-%28u-o%5E2+%2F2%29T%29%5E2+%2F
    #+%282*%28o%5E2+T%29+%29%29%2F%28sqrt%282pi*T%29*o%29%29
    
    local_likelyhood(vec) = log_likelyhood(valid_logdiff, delta_times_in_year, vec[1],vec[2])

    #just guessed paramters, base on past 15 years or sor

    result = optimize(local_likelyhood, [0.02, 0.15])

    params = Optim.minimizer(result)

    return params[1], params[2]
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

function lognormal(S::Float64,o::Float64,mu::Float64)
    y = 1/(S * sqrt(2pi*o^2)) * exp(-(log(S)-mu)^2 / (2 * o^2))
    return y
end



#conditions reminiscent of the past two decade for SNP 500 - brute-guessed numbers
#test = BlackScholesModel(0.02,300,20000,0.14,0.04,0.025)

#payouts, mwcloss, cut_index = evaluate_put_position_BS(1025.0,test,1000.0,20,0.95)

#hist = histogram(payouts, color = "green", xlabel = "Payout", 
#ylabel = "Simulated frequencies of payouts", title = "CVar visualization", 
#label = "Simulated payoffs and losses", normalize = :pdf, dpi = 1200)
#vline!(hist,[payouts[cut_index+1]], ls = :dash, color = "red", label = "Confidence cut at 95%")
#vline!(hist, [mwcloss], ls = :dash, color = "blue", label = "Mean loss beyond confidence cut")

#t = LinRange(minimum(payouts),maximum(payouts),1000)
#t_shift = -minimum(payouts) + 0.00001
#o = std(payouts)
#mu = mean(payouts)

#lgn_params = [log(mu^2/sqrt(mu^2 +o^2)), sqrt(log(1+o^2 / mu^2))]

#y = @. 1/sqrt(2pi * o^2) * exp(-(t-mu)^2/(2o^2))
#plot!(hist, t, y, label = "Fitted normal distribution", color = "orange", linewidth = 1.0)



#simulated_movements = batch_simulator(test,1000.0,20)
#number_to_display = 100
#colors = [RGB(0.1,1 - n/number_to_display,0.5*n/number_to_display) for n in 1:number_to_display]'
#times = [i*test.dt for i in 1:test.NTimeSteps]
#plot0 = plot(times, [simulated_movements[n,:] for n in 1:number_to_display], 
#title = "Stock Price under BS model", legend = false, 
#linewidth = 0.5, linecolors = colors, xlabel = "Time", ylabel = "Possible future price", dpi = 500)


#OTP analysis
mu_otp, sigma_otp = estimate_BS_params(OTP_DATA_DF)

#set up simulator 


#since our dt is set to be constant, we need to "cheat"

local_zero_date = date_to_seconds(OTP_DATA_DF[1,"Date"])
local_end_date = date_to_seconds(OTP_DATA_DF[end,"Date"])
Number_of_years = (local_end_date - local_zero_date) / (3600 * 24 * 365)
Number_of_days = (local_end_date - local_zero_date) / (3600 * 24)



dt_OTP = Number_of_years/Number_of_days #simulate "daily" prices - do not deal with gaps due to trading stops

N_OTP = 10000
#the historic interest rate is not of interest here, but it was around 2% on average
OTP_MODEL = BlackScholesModel(dt_OTP, Number_of_days, N_OTP, sigma_otp, mu_otp, 0.02)

#TODO: Unmess this line
times_in_seconds = collect(LinRange(local_zero_date, local_end_date, Int64(Number_of_days)))
OTP_PLOT_TIMES = @. Date(unix2datetime(times_in_seconds))

plot0 = plot(OTP_DATA_DF[!,"Date"], OTP_DATA_DF[!,"Close"], xlabel = "Date", ylabel = "Closing Price [EUR]", 
title = "BS Analysis of 'OTP Nyrt' Stock Prices", color = "red", label = "Historical Stock Price", dpi = 900)

#AUX
#test0 = BlackScholesModel(0.02,60,200000,0.14,0.04,0.025)
#final_prices = simulate_system_no_track(test0,100.0)[:,2]

#o = std(final_prices)
#mu = mean(final_prices)

#lgn_params = [log(mu^2/sqrt(mu^2 +o^2)), sqrt(log(1+o^2 / mu^2))]

#t = LinRange(minimum(final_prices),maximum(final_prices),1000)

#y = @. 1/sqrt(2pi * o^2) * exp(-(t-mu)^2/(2o^2))
#y2 = lognormal.(t, lgn_params[2], lgn_params[1])

#hist0 = histogram(final_prices, color = "green", xlabel = "Stock price", bins = 200,
#ylabel = "Simulated frequencies of prices", title = "Final Simulated Stock Price distribution", 
#label = L"Stock Prices $\mu = 0.04$, $\sigma = 0.14$", normalize = :pdf, dpi = 1200)

#plot!(hist0, t, y, label = "Fitted normal distribution", color = "orange", linewidth = 1.0)
#plot!(hist0, t, y2, label = "Fitted lognormal distribution", color = "blue", linewidth = 1.0)


