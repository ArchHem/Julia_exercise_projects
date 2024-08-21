module HesParameterEst
include("./heston_model.jl")
using Turing, Statistics, Optim, ForwardDiff

function heston_joint_LLM(Q::Vector,V::Vector,param_vec::Vector)
    
    r, k, theta, sigma, rho = param_vec

    start_index = 2 #needs to be at least 2 

    #deal with singularity at zeroth element
    Qp = @views Q[start_index+1:end]
    
    Vp = @views V[start_index+1:end]
    Vn = @views V[start_index:end-1]

    #neglect constant terms
    to_sum_1 = @. -log(sigma)- log(Vn) - 0.5 * log(1-rho^2)
    to_sum_2 = @. -(Qp-1.0-r)^2 / (2*Vn*(1-rho^2)) + rho*(Qp-1.0-r)*(Vp - Vn- theta*k + k*Vn) / (Vn*sigma*(1-rho^2))
    to_sum_3 = @. -( Vp-Vn-theta*k + k*Vn)^2 / (2*sigma^2 *Vn*(1-rho^2))
    to_sum = @. to_sum_1 + to_sum_2 + to_sum_3

    weight = -sum(to_sum)
    return weight
end

function HestonEstimator(S_data::Vector{T},x0::Vector{T}) where T<:Real
    #performs the estimation explained in:
    #https://www.valpo.edu/mathematics-statistics/files/2015/07/Estimating-Option-Prices-with-Heston’s-Stochastic-Volatility-Model.pdf

    Q = @. S_data[2:end]/S_data[1:end-1] #return ratio
    N = length(Q)

    #around middle of pg 7
    V_est = @views [var(Q[1:i]) for i in 1:N]
    #manually overwrite first element to be zero
    V_est[1] = zero(T)

    NLL(params) = heston_joint_LLM(Q,V_est,params)
    NLL_deriv(params) = ForwardDiff.gradient(NLL,params)

    lower = [-Inf, zero(T), zero(T), zero(T), -one(T)]
    higher = [Inf, Inf, Inf, Inf, one(T)]
    results = optimize(NLL, lower,higher,x0,Fminbox(GradientDescent()), inplace = false, autodiff = :forward)

    return results, Q, V_est
end
export HestonEstimator
end

using .HesParameterEst, CSV, DataFrames, Optim, Plots

const S_data = CSV.read("fn_simulations/fn_data/goog_closing_prices_2008_2018.csv",delim = ",", DataFrame)
const S0 = collect(S_data[!,"Close"])

const x0 = [0.06/252,0.02,0.04/252,0.1/sqrt(252),-0.4]
const opt, Q, V = HestonEstimator(S0,x0)
parameters = Optim.minimizer(opt)

#we might want to upscale our parameters back to the anual basis: can be done via dimensional analysis and by multipling w number of trading days