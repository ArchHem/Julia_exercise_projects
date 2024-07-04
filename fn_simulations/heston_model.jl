module Heston_support
using ProgressBars, StaticArrays, Plots, Statistics, CSV, DataFrames, Statistics, Dates, Optim, LaTeXStrings, SpecialFunctions, Roots

struct HestonModel
    dt::Float64
    NTimeSteps::Int64
    NSituationsBatch::Int64
    #long - twem avg. volaritity
    theta::Float64
    #drift rate
    mu::Float64
    #interest rate of underyling, risk-free bond
    r::Float64
    #volatility of volatility
    epsilon::Float64
    #rate of return to 'normal' volatility
    k::Float64
    #correlation coeff
    p::Float64
end

function ReLu(x)
    return max(0.,x)
end

function evolve_heston_prices(system::HestonModel,S,V,Z_s)
    S_new = S * exp((system.r - 0.5*V)*system.dt + sqrt(system.dt*V)*Z_s)

    return S_new
end

function evolve_heston_volats(system::HestonModel,S,V,Z_v)
    V_new = V + system.k*(system.theta-V)*system.dt + system.epsilon * sqrt(system.dt*V)*Z_v
    return V_new
end

function evolve_heston_batch(system::HestonModel,S0::Float64,V0::Float64)
    N_systems = system.NSituationsBatch
    N_timesteps = system.NTimeSteps

    S = S0*ones(N_systems)
    V = V0*ones(N_systems)

    p = system.p

    for i in 1:N_timesteps
        Z_v = randn(N_systems)
        Z_s = p*Z_v + randn(N_systems)*sqrt(1-p^2)
        
        #fix the dotting errors and keep some memory
        S_new = evolve_heston_prices.(Ref(system),S,V,Z_s)
        V_new = evolve_heston_volats.(Ref(system),S,V,Z_v)
        @. S = S_new
        #apply corrections
        @. V = ReLu(V_new)
    end

    return S, V
end

function evolve_heston_full(system::HestonModel,S0::Float64,V0::Float64,NBatches::Int64)
    N_systems = system.NSituationsBatch

    S = S0*ones(N_systems*NBatches)
    V = V0*ones(N_systems*NBatches)

    maxcount = NBatches-1
    Threads.@threads for k in ProgressBar(0:maxcount)

        local_prices, local_volats = evolve_heston_batch(system,S0,V0)
        
        
        @. S[k*N_systems+1:(k+1)*N_systems] = local_prices
        @. V[k*N_systems+1:(k+1)*N_systems] = local_volats

    end

    return S, V
end


#dirty metaprogramming trick if we want to export everything!

#exported_names = names(@__MODULE__, all=true) .|> Symbol

#eval(Expr(:export, exported_names...))

#module end 


end








