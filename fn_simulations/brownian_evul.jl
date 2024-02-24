using ProgressBars, StaticArrays, Plots, Statistics

struct BrownianMotion
    NScenarios::Int64
    NTimeSteps::Int64
    dt::Float64
    sigma::Float64
end

function integrate_brownian_motions(input::BrownianMotion)

    time_array = collect(1:input.NTimeSteps)
    lvalues = zeros(Float64,input.NScenarios)
    total_dat = zeros(Float64,(input.NScenarios,input.NTimeSteps))
    for t in ProgressBar(1:input.NTimeSteps)
        total_dat[:,t] = lvalues
        lvalues += sqrt(input.dt) * input.sigma * randn(Float64,input.NScenarios)
        
    end

    return time_array, total_dat
end

example_brownian_sim = BrownianMotion(50000,3600,1.0,1.0)
times, values = integrate_brownian_motions(example_brownian_sim)

N_examps = 30
colors = [RGB(0.1,1 - n/N_examps,0.5*n/N_examps) for n in 1:N_examps]'
final_pos = values[:,length(times)]
mu = mean(final_pos)
sigma = std(final_pos)
skewness = sum(((final_pos .-mu)./sigma).^3)/length(final_pos)
curtosis = sum(((final_pos .-mu)./sigma).^4)/length(final_pos)

println("Measured position mean: ",mu)
println("We expected a position mean of 0.0.")

println("Measured position std: ", sigma)
println("We expected a sigma of ", sqrt(example_brownian_sim.dt * example_brownian_sim.NTimeSteps))


println("Measured position skewness: ", skewness)
println("We expect a skewness of 0.0")
println("Measured position curtosis: ", curtosis)
println("We expect a curtosis of ", 3*example_brownian_sim.sigma^2)

#plot0 = plot(times, [values[n,:] for n in 1:N_examps], 
#title = "Brownian movement examples", legend = false, 
#linewidth = 0.5, linecolors = colors, xlabel = "Timestep", ylabel = "Position", dpi = 500)

plot1 = plot(histogram(final_pos, color = "green", title = "Final position histogram",
legend = false, xlabel = "Position", ylabel = "Position frequencies", bins = 100, dpi = 500, normalize = :pdf, label = ""))

xplot = LinRange(-250,250,1000)
sigmaplot = sqrt(example_brownian_sim.dt * example_brownian_sim.NTimeSteps)
yplot = @. exp(-xplot^2/(2*sigmaplot^2))/sqrt(2pi * sigmaplot^2)

plot!(plot1,xplot,yplot, color = "red",ls =:dash, label = "Expected normal distribution", legend = true, linewidth = 2.0)





