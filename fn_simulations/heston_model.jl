module Heston_support
using ProgressBars, Optim, SpecialFunctions, Roots, FFTW

export HestonModel, evolve_heston_batch, evolve_heston_full, eval_call_option

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

function eval_call_option(S_current::Float64,V_current::Float64,K::Float64, model::HestonModel, lambda::Float64,
    max_freq::Float64 = 1000.0,N_freqs::Int64 = 2^10)
    #lambda: coefficient of the linear function of the market cost for the volatility

    rho = model.p
    r = model.r
    sigma = model.epsilon
    k = model.k
    theta = model.theta
    T = model.dt*model.NTimeSteps

    u1 = 0.5
    u2 = -0.5

    a = k*theta
    b1 = k + lambda - rho*sigma
    b2 = k + lambda

    d1(phi) = sqrt((im*rho*sigma*phi-b1)^2 -sigma^2 * (2*u1*im*phi  - phi^2))
    d2(phi) = sqrt((im*rho*sigma*phi-b2)^2 -sigma^2 * (2*u2*im*phi  - phi^2))

    g1(phi) = (b1-rho*sigma*im*phi+d1(phi)) / (b1-rho*sigma*im*phi - d1(phi))
    g2(phi) = (b2-rho*sigma*im*phi+d2(phi)) / (b2-rho*sigma*im*phi - d2(phi))

    D1(phi) = (b1-rho*sigma*im*phi + d1(phi)) / sigma^2 * (1-exp(d1(phi)*T))/(1-g1(phi)*exp(d1(phi)*T))
    D2(phi) = (b2-rho*sigma*im*phi + d2(phi)) / sigma^2 * (1-exp(d2(phi)*T))/(1-g2(phi)*exp(d2(phi)*T))

    C1(phi) = r*im*phi*T + a / sigma^2 * ((b1-rho*im*sigma*phi + d1(phi))*T -2 * log((1-g1(phi)*exp(d1(phi)*T))/(1-g1(phi))))
    C2(phi) = r*im*phi*T + a / sigma^2 * ((b2-rho*im*sigma*phi + d2(phi))*T -2 * log((1-g2(phi)*exp(d2(phi)*T))/(1-g2(phi))))

    x = log(S_current)
    f1(phi) = exp(C1(phi)+D1(phi)*V_current + im*phi*x)
    f2(phi) = exp(C2(phi)+D2(phi)*V_current + im*phi*x)

    d_phi = max_freq / N_freqs

    #avoid singalirity

    phi_range = LinRange(d_phi,max_freq+d_phi,N_freqs)

    f1_arr = f1.(phi_range)
    f2_arr = f2.(phi_range)

    return f1_arr, f2_arr

end

#dirty metaprogramming trick if we want to export everything!

#exported_names = names(@__MODULE__, all=true) .|> Symbol

#eval(Expr(:export, exported_names...))

#module end 

end








