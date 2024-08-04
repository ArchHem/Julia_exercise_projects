using ProgressBars, Plots, Statistics, Statistics, Dates, Optim, LaTeXStrings, SpecialFunctions, Roots, FFTW
#leads to namespace errors!

include("./heston_model.jl")
using .Heston_support

T = 2.0
S0 = 100.0
V0 = 0.2^2
K0 = 30.0
theta = 0.2^2
eps = 0.3
alpha = 1.5
k = 3.0

#implicit condition of 2k*theta > eps^2

cond = 2*k*theta - eps^2

local_model_1 = HestonModel(0.001,2000,100000,theta,0.02,0.02,eps,k,-0.6)
logKs1, Cs1 = get_call_vals(S0,V0,T,K0,alpha,local_model_1,N = 12,du = 0.1)

log_moniness1 = log(S0).-logKs1

plotinst = plot(log_moniness1, Cs1, color = "green", 
label = L"$C_{\tau = 2.0}(\alpha = 1.5, \rho = -0.6)$ - FFT", xlabel = L"$M = \log{(S_0 / K)}$", dpi = 1200,
xlim = [-1,1], ylabel = L"$C_{\tau}(M)$")

local_model_2 = HestonModel(0.001,2000,100000,theta,0.02,0.02,eps,k,0.6)
logKs2, Cs2 = get_call_vals(S0,V0,T,K0,alpha,local_model_2,N = 12,du = 0.1)

log_moniness2 = log(S0).-logKs2

plot!(log_moniness2, Cs2, color = "red", 
label = L"$C_{\tau = 2.0}(\alpha = 1.5, \rho = 0.6)$ - FFT", xlabel = L"$M = \log{(S_0 / K)}$", dpi = 1200,
xlim = [-1,1], ylabel = L"$C_{\tau}(M)$")

#replicate same scenario via MC method

valid_indeces = abs.(log_moniness1) .< 1
valid_moniness = log_moniness1[valid_indeces]
logputs_valid = logKs1[valid_indeces]
k_valid = exp.(logputs_valid)

#generate mc price


function get_mc_calls(model::HestonModel,K,S0,V0;NBatches = 8)
    prices, volats = evolve_heston_full(model,S0,V0,NBatches)
    #maybe time along which axis the bellow is faster? Assume mean() is better operating columnwise!
    discount = exp(-model.r*model.dt*model.NTimeSteps)
    payoffs = vec(discount*mean(ReLu.(prices .- K'), dims = 1))
    
    return payoffs
end

mc_prices_1 = get_mc_calls(local_model_1,k_valid,S0,V0)
mc_prices_2 = get_mc_calls(local_model_2,k_valid,S0,V0)


scatter!(valid_moniness,mc_prices_1,mc = "orange", label = L"$C_{\tau} ,\rho = -0.6$ - MC", markersize = 1.5,  markerstrokewidth = 0.0)

scatter!(valid_moniness,mc_prices_2,mc = "blue", label = L"$C_{\tau} ,\rho = 0.6$ - MC", markersize = 1.5,  markerstrokewidth = 0.0)