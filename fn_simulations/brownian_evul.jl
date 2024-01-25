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

example_brownian_sim = BrownianMotion(50000,3600,0.1,1.0)
times, values = integrate_brownian_motions(example_brownian_sim)

N_examps = 30
colors = [RGB(0.1,1 - n/N_examps,0.5*n/N_examps) for n in 1:N_examps]'
final_pos = values[:,length(times)]
mu = mean(final_pos)
sigma = std(final_pos)
skewness = sum(((final_pos .-mu)./sigma).^3)/length(final_pos)
curtosis = sum(((final_pos .-mu)./sigma).^4)/length(final_pos)

println("Measured position mean: ",mu)
println("Measured position std: ", sigma)
println("Measured position skewness: ", skewness)
println("Measured position curtosis: ", curtosis)

#plot0 = plot(times, [values[n,:] for n in 1:N_examps], 
#title = "Brownian movement examples", legend = false, 
#linewidth = 0.5, linecolors = colors, xlabel = "Timestep", ylabel = "Position", dpi = 500)

plot1 = plot(histogram(final_pos, color = "green", title = "Final position histogram",
legend = false, xlabel = "Position", ylabel = "Position counts", bins = 100, dpi = 500))

