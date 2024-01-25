using ProgressBars, StaticArrays, Plots, Statistics

struct BlackScholesModel
    dt::Float64
    NTimeSteps::Int64
    NSituations::Int64
    #volatitility
    sigma::Float64
    #drift rate
    mu::Float64
end

function SimulateBS(input::BlackScholesModel)

    logsteps = zeros(Float64,(input.NSituations, input.NTimeSteps))
    for t in 1:input.NTimeSteps

    end
end