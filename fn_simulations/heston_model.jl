module Heston_support
using ProgressBars, Optim, SpecialFunctions, Roots, FFTW

export HestonModel, evolve_heston_batch, evolve_heston_full, heston_characeristic, get_call_vals, ReLu

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

function heston_characeristic(phi,S,V,tau,model::HestonModel)

    x = log(S)
    sigma = model.epsilon
    mu = model.mu
    a = model.k
    b = model.theta
    rho = model.p

    gamma = sqrt(sigma^2 * (phi^2 + im*phi) + (a-im*sigma*phi*rho)^2)

    sec_term = (im*phi+phi^2)*V / (gamma*coth(gamma*tau/2) + (a-im*rho*sigma*phi))
    exp_in = im*phi*x - sec_term + a*b*(a-im*rho*sigma*phi)*tau / sigma^2 + im*phi*mu*tau

    power = 2*a*b/sigma^2
    denom = cosh(gamma*tau/2) + (a-im*rho*sigma*phi) / gamma * sinh(gamma*tau / 2)

    res = exp(exp_in)/denom^power

    return res

end

function get_call_vals(S0,V0,tau,K,alpha,model::HestonModel; N::Int64 = 10,du::Float64)

    true_N = 2^N
    phifreqs = LinRange(0,true_N*du,true_N)
    dlogK = 2 * pi / (true_N*du)

    logKs = log(K) .+ LinRange(0,true_N*dlogK,true_N)

    discount = exp(-model.r*tau)

    damped_char_function = heston_characeristic.(phifreqs .-((alpha+1)*im),S0,V0,tau,Ref(model))

    denom = @. ((alpha + 1*im * phifreqs) * (alpha + 1 + 1*im * phifreqs))

    quant = @. damped_char_function / denom

    omega = du .* ones(true_N)

    omega[1] = du/2

    x_vals = @. exp(-1*im * log(K) * phifreqs) * discount * quant * omega
    y_vals = fft(x_vals)

    corrector = exp.(-alpha * logKs) / pi

    call_vals = @. corrector*real(y_vals)

    return logKs, call_vals

end
#dirty metaprogramming trick if we want to export everything!

#exported_names = names(@__MODULE__, all=true) .|> Symbol

#eval(Expr(:export, exported_names...))

#module end 

end








