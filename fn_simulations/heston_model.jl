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

    


#display effects of 'tilt'

N_sits = 7
ps = LinRange(-0.9,0.9,N_sits)
colors = [RGB(0.1,i/N_sits,1-i/N_sits) for i in 1:N_sits]

NperBatch = 160000
nbins = LinRange(0.0,400,100)

result_prices = Vector{Vector{Float64}}(undef,(N_sits,))

for o in 1:N_sits

    local examp = HestonModel(0.01,1000,NperBatch,0.25^2,0.08,0.05,0.3,3.0,ps[o])

    local S, V = evolve_heston_full(examp,100.0,0.3^2,8)
    
    lstring = string(round(ps[o]; sigdigits=3))
    if o == 1
        global X = stephist(S,color = colors[o], 
        xlim = [0,400], normalize=:pdf,
        label = L"$\rho =$"*lstring, bins = nbins, xlabel = "S", ylabel = "P(S)",
        dpi = 1600)
    else
        stephist!(X,S,color = colors[o], xlim = [0,400], 
        normalize=:pdf, label = L"$\rho =$"*lstring, bins = nbins)
    end
end

display(X)








