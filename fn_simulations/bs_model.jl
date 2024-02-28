using ProgressBars, StaticArrays, Plots, Statistics

struct BlackScholesModel
    dt::Float64
    NTimeSteps::Int64
    NSituations::Int64
    #volatitility
    sigma::Float64
    #drift rate: analogous to underlying mean interest
    mu::Float64
end
