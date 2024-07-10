using ProgressBars, Plots, Statistics, Statistics, Dates, Optim, LaTeXStrings, SpecialFunctions, Roots, FFTW
#leads to namespace errors!

include("./heston_model.jl")
using .Heston_support: HestonModel, evolve_heston_batch, evolve_heston_full


plotlyjs() 

function setfield(model::HestonModel, field::Symbol, value)
    fields = collect(fieldnames(typeof(model)))
    field_vals = collect(getfield.(Ref(model),fields))
    
    @. field_vals[fields == field] = value
    new_model = HestonModel(field_vals...)
    return new_model
end

#rho dependence check
#=
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
=#


#option pricing 1

#=
T = 10
N_t = 1000
N_sits = 20000
dt = T/N_t
local_model = HestonModel(dt,100,N_sits,0.2^2,0.04,0.02,0.6,3.0,-0.6)

N = 40
S0 = LinRange(80.0,120.0,N) .* ones(N)'
K0  = LinRange(60.0,120.0,N)' .* ones(N)

#use parallelism here 

function S_K_price(S0,K0,T,V0,model::HestonModel)

    call_results = Matrix{Float64}(undef,size(S0))
    put_results = Matrix{Float64}(undef,size(S0))
    Threads.@threads for k in eachindex(S0)
        #technically as long as K is part of the double loop, its inefficent, could factor it out
        local_prices, local_volats = evolve_heston_batch(model,S0[k],V0)
        deval = exp(-model.r*T)
        call_gains = deval * mean(max.(local_prices.-K0[k],0.0))
        put_gains = deval * mean(max.(K0[k] .-local_prices,0.0))
        call_results[k] = call_gains
        put_results[k] = put_gains
    end
    return call_results, put_results
end

call_res, put_res = S_K_price(S0,K0,T,0.25^2,local_model)

scatter(S0,K0,put_res, color = "green", markersize = 0.8, xlabel = "S(0)", ylabel = "K",
 zlabel = "U(0)", label = nothing)
 =#


#V0-K surface

#=
T = 10
N_t = 1000
N_sits = 20000
dt = T/N_t
local_model = HestonModel(dt,100,N_sits,0.2^2,0.04,0.02,0.6,3.0,-0.6)

N = 40
V0 = LinRange(0.1^2,0.5^2,N) .* ones(N)'
K0  = LinRange(60.0,120.0,N)' .* ones(N)

function V_K_price(V0,K0,T,S0,model::HestonModel)

    call_results = Matrix{Float64}(undef,size(V0))
    put_results = Matrix{Float64}(undef,size(V0))
    Threads.@threads for k in eachindex(V0)
        local_prices, local_volats = evolve_heston_batch(model,S0,V0[k])
        deval = exp(-model.r*T)
        call_gains = deval * mean(max.(local_prices.-K0[k],0.0))
        
        put_gains = deval * mean(max.(K0[k] .-local_prices,0.0))
        call_results[k] = call_gains
        put_results[k] = put_gains
    end
    return call_results, put_results
end

call_res, put_res = V_K_price(V0,K0,T,100.0,local_model)

scatter(V0,K0,call_res, color = "green", markersize = 0.8, xlabel = "V(0)", ylabel = "K",
 zlabel = "U(0)", label = nothing)
=#

#surface as function of K and xi

#=
T = 10
N_t = 1000
N_sits = 40000
dt = T/N_t


N = 40
Xi0 = LinRange(0.0,3.0,N) .* ones(N)'
K0  = LinRange(60.0,120.0,N)' .* ones(N)

S0 = 100.0
V0 = 0.25^2

function Xi_K_price(Xi0,K0,T,S0,V0,model::HestonModel)

    call_results = Matrix{Float64}(undef,size(Xi0))
    put_results = Matrix{Float64}(undef,size(Xi0))
    Threads.@threads for k in ProgressBar(eachindex(Xi0))
        local_model = setfield(model,:epsilon,Xi0[k])
        local_prices, local_volats = evolve_heston_batch(local_model,S0,V0)
        deval = exp(-model.r*T)
        call_gains = deval * mean(max.(local_prices.-K0[k],0.0))
        
        put_gains = deval * mean(max.(K0[k] .-local_prices,0.0))
        call_results[k] = call_gains
        put_results[k] = put_gains
    end
    return call_results, put_results
end

local_model = HestonModel(dt,100,N_sits,0.2^2,0.04,0.02,0.6,3.0,-0.6)

call_res, put_res = Xi_K_price(Xi0,K0,T,100.0,V0,local_model)


scatter(Xi0,K0,put_res, color = "green", markersize = 0.8, xlabel = "Xi", ylabel = "K",
 zlabel = "U(0)", label = nothing)

 =#

 #surface as function of theta and K

T = 10
N_t = 1000
N_sits = 40000
dt = T/N_t


N = 40
Theta0 = LinRange(0.05^2,0.8^2,N) .* ones(N)'
K0  = LinRange(60.0,120.0,N)' .* ones(N)

S0 = 100.0
V0 = 0.25^2

function theta_K_price(theta0,K0,T,S0,model::HestonModel)

    call_results = Matrix{Float64}(undef,size(theta0))
    put_results = Matrix{Float64}(undef,size(theta0))
    Threads.@threads for k in ProgressBar(eachindex(theta0))
        local_model = setfield(model,:theta,theta0[k])
        local_prices, local_volats = evolve_heston_batch(local_model,S0,theta0[k])
        deval = exp(-model.r*T)
        call_gains = deval * mean(max.(local_prices.-K0[k],0.0))
        
        put_gains = deval * mean(max.(K0[k] .-local_prices,0.0))
        call_results[k] = call_gains
        put_results[k] = put_gains
    end
    return call_results, put_results
end

local_model = HestonModel(dt,100,N_sits,0.2^2,0.04,0.02,0.6,1.0,-0.6)

call_res, put_res = theta_K_price(Theta0,K0,T,100.0,local_model)


scatter(Theta0,K0,put_res, color = "green", markersize = 0.8, xlabel = "Theta", ylabel = "K",
 zlabel = "U(0)", label = nothing)