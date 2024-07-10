using ProgressBars, Plots, Statistics, Statistics, Dates, Optim, LaTeXStrings, SpecialFunctions, Roots, FFTW
#leads to namespace errors!

include("./heston_model.jl")
using .Heston_support

local_model = HestonModel(0.01,100,1000.0,0.2^2,0.04,0.02,0.6,3.0,-0.6)
f1, f2 = eval_call_option(100.0,0.2^2,120.0,local_model,0.0,100.0,2^12)


plot(LinRange(0,100.0,2^12),real.(f1))


