include("./heston_model.jl")
using ProgressBars, StaticArrays, Plots, Statistics, CSV, DataFrames, Statistics, Dates, Optim, LaTeXStrings, SpecialFunctions, Roots

using .Heston_support

N_sits = 7
ps = LinRange(-0.9,0.9,N_sits)
colors = [RGB(0.1,i/N_sits,1-i/N_sits) for i in 1:N_sits]

NperBatch = 160000
nbins = LinRange(0.0,200,200)

result_prices = Vector{Vector{Float64}}(undef,(N_sits,))

for o in 1:N_sits

    local examp = HestonModel(0.01,100,NperBatch,0.2^2,0.02,0.02,0.6,3.0,ps[o])

    local S, V = evolve_heston_full(examp,100.0,0.25^2,8)
    
    lstring = string(round(ps[o]; sigdigits=3))
    if o == 1
        global X = stephist(S,color = colors[o], 
        xlim = [0,200], normalize=:pdf,
        label = L"$\rho =$"*lstring, bins = nbins, xlabel = "S", ylabel = "P(S)",
        dpi = 1600)
    else
        stephist!(X,S,color = colors[o], xlim = [0,200], 
        normalize=:pdf, label = L"$\rho =$"*lstring, bins = nbins)
    end
end

display(X)