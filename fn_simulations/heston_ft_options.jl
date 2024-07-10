using ProgressBars, Plots, Statistics, Statistics, Dates, Optim, LaTeXStrings, SpecialFunctions, Roots, FFTW
#leads to namespace errors!

include("./heston_model.jl")
using .Heston_support

local_model = HestonModel(0.01,100,1000.0,0.2^2,0.02,0.02,0.6,3.0,-0.6)
S0 = 100.0
logKs, Cs = get_call_vals(S0,0.2^2,2.,5.0,1.5,local_model,N = 12,du = 0.1)

log_moniness = log(S0).-logKs

plot(log_moniness, Cs, color = "green", 
label = L"$C_{\tau = 1.0}(\alpha = 1.5, \rho = -0.6)$ - FFT", xlabel = L"$M = \log{(S_0 / K)}$", dpi = 1200,
xlim = [-1,1], ylabel = L"$C_{\tau}(M)$")

local_model = HestonModel(0.01,100,1000.0,0.2^2,0.02,0.02,0.6,3.0,0.6)
logKs, Cs = get_call_vals(S0,0.2^2,2.,5.0,1.5,local_model,N = 12,du = 0.1)

log_moniness = log(S0).-logKs

plot!(log_moniness, Cs, color = "red", 
label = L"$C_{\tau = 1.0}(\alpha = 1.5, \rho = 0.6)$ - FFT", xlabel = L"$M = \log{(S_0 / K)}$", dpi = 1200,
xlim = [-1,1], ylabel = L"$C_{\tau}(M)$")
