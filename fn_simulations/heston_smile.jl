include("./heston_model.jl")
using .Heston_support: HestonModel, evolve_heston_batch, evolve_heston_full
using Optim, Plots, Statistics, Distributions, SpecialFunctions, Roots, ProgressBars

const N_situ = 100000
const S0 = 100.0
const rho = -0.5
model = HestonModel(0.001,500,N_situ,0.2^2,0.05,0.02,0.6,3.0,rho)
const S, V = evolve_heston_full(model,S0,0.2^2,5)

function normal_cdf(x)
    z = (1 + erf(x/sqrt(2)))/2
    return z
end

function ReLu(x::T) where T<:Real
    outp = x > zero(T) ? x : zero(T)
    return outp
end

function BS_fair_price_call(sigma,r,asset_price,strike_price,time_to_expiry)
    #Assumes that the time to expiry is goven in terms of yearly amount
    d_p = 1/(sigma*sqrt(time_to_expiry)) * (log(asset_price / strike_price) + (r + sigma^2 / 2)*time_to_expiry)
    d_m = d_p - sigma * sqrt(time_to_expiry)

    C = normal_cdf(d_p)*asset_price - strike_price*normal_cdf(d_m)*exp(-r*time_to_expiry)
    return C
end 

function HestonCallPrice(prices::Vector,K::Vector,r::C,T::C) where C<:Real
    N = length(K)
    fair_prices = exp(-r*T) .* [mean(ReLu.(prices .- K[i])) for i in 1:N]
    return fair_prices
end



const r = model.r
const T = model.dt * model.NTimeSteps
const asset_price = mean(S)

const K = collect(LinRange(50.,150.,200))
const call_prices = HestonCallPrice(S,K,r,T)



N = length(K)

x0 = 0.2
res = zeros(N)
for i in ProgressBar(1:N)
    target_func(sigma) = BS_fair_price_call(sigma,r,S0,K[i],T)-call_prices[i]
    root = find_zero(target_func,x0)
    global x0 = root
    res[i] = root
end


x = plot(K, res, xlabel = "Strike Price", ylabel = "BS Implied Vol.", 
title = "Heston MC run, with ρ = $(round(rho,digits = 2))",color = "red",
label = "σ", dpi = 1200)
vline!(x,[asset_price*exp(-T*r)], color = "blue", label = "Discounted asset price", ls = :dash)





