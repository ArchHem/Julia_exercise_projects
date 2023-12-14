function sigmoid(x::Vector{Float64})
    quant =  1.0 ./ (1.0 .+ exp.(.-x))
    return quant
end
    


mutable struct NN_network
    N_neurons::Vector{Int64}
    layer_weights::Vector{Matrix{Float64}}
    layer_biases::Vector{Vector{Float64}}
    act_functions::Vector{Function}
    function NN_network(N_neurons::Vector{Int64},act_functions::Vector{Function})
        layer_weights = Vector{Matrix{Float64}}(undef,length(N_neurons)-1)
        layer_biases = Vector{Vector{Float64}}(undef,length(N_neurons)-1)
        for i in 1:length(N_neurons)-1
            layer_weights[i] = rand(Float64,(N_neurons[i+1],N_neurons[i]))
            layer_biases[i] = rand(Float64,(N_neurons[i+1],))
        end
        new(N_neurons,layer_weights,layer_biases,act_functions)
    end
end 

neuron_arr = Vector{Int64}([3,4,2,1])
funcs = Vector{Function}([sigmoid,sigmoid,sigmoid,sigmoid])
test = NN_network(neuron_arr,funcs)

println(size(test.layer_weights[1]))
println(size(test.layer_biases[1]))