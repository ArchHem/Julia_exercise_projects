function sigmoid(x::Vector{Float64})
    quant =  1.0 ./ (1.0 .+ exp.(.-x))
    return quant
end

function sigmoid_der(x::Vector{Float64})
    quant = sigmoid(x) .* (1.0 .- sigmoid(x))
    return quant
end

function binary_crossentropy(ypredict::Vector{Float64},
    ylabel::Vector{Float64})
    quant = -ypredict .* log.(ylabel) - (1.0 .- ylabel) .* log.(1.0 .- ypredict)
    return quant[1]
end

function binary_crossentropy_der(ypredict::Vector{Float64},
    ylabel::Vector{Float64})
    quant = - log.(ylabel) - (1 .-ylabel) .* (-1.0) ./ (1.0 .-ypredict)
    return quant[1]
end

struct NN_network
    N_neurons::Vector{Int64}
    layer_weights::Vector{Matrix{Float64}}
    layer_biases::Vector{Vector{Float64}}
    act_functions::Vector{Function}
    act_function_der::Vector{Function}
    loss_function::Function
    loss_function_der::Function
    function NN_network(N_neurons::Vector{Int64},act_functions::Vector{Function},act_function_der::Vector{Function},
        loss_function::Function,loss_function_der::Function)
        layer_weights = Vector{Matrix{Float64}}(undef,length(N_neurons)-1)
        layer_biases = Vector{Vector{Float64}}(undef,length(N_neurons)-1)
        for i in 1:length(N_neurons)-1
            layer_weights[i] = rand(Float64,(N_neurons[i+1],N_neurons[i]))
            layer_biases[i] = rand(Float64,(N_neurons[i+1],))
        end
        new(N_neurons,layer_weights,layer_biases,act_functions,act_function_der,loss_function,loss_function_der)
    end
end 

function modify_layer(layer_index::Int64,new_weights::Matrix{Float64},new_biases::Vector{Float64},NN_instance::NN_network)
    NN_instance.layer_weights[layer_index] .= new_weights
    NN_instance.layer_biases[layer_index] .= new_biases
end

function apply_layer(input_vector::Vector{Float64},NN_instance::NN_network,index::Int64)
    l_weights = NN_instance.layer_weights[index,]
    l_bias = NN_instance.layer_biases[index,]
    l_afunc = NN_instance.act_functions[index,]
    zvec = l_weights * input_vector + l_bias
    output_vector = l_afunc(zvec)
    return output_vector, zvec
end

function propegate_thr_network(input_vector::Vector{Float64},NN_instance::NN_network)
    number_of_layers = length(NN_instance.N_neurons)
    output_vector = input_vector
    for indices in 1:number_of_layers-1
        output_vector, zvec = apply_layer(output_vector,NN_instance,indices)
    end
    return output_vector
end

function evaluate_loss(input_vector::Vector{Float64},label::Vector{Float64},NN_instance::NN_network)
    output = propegate_thr_network(input_vector,NN_instance)
    loss = NN_instance.loss_function(output,label)
    return loss
end

function calculate_activations(input_vector::Vector{Float64},NN_instance::NN_network)
    N = length(NN_instance.N_neurons)
    activations = Vector{Vector{Float64}}(undef,(N-1,))
    zvecs = Vector{Vector{Float64}}(undef,(N-1,))
    temp = input_vector
    for i in 1:N-1
        temp, lzvec = apply_layer(temp,NN_instance,i)
        activations[i] = temp
        zvecs[i] = lzvec
    end
    return activations, zvecs
end 

function backpropegate(input_vector::Vector{Float64},label::Vector{Float64},NN_instance::NN_network)
    activations, zvecs = calculate_activations(input_vector,NN_instance)
    N = length(NN_instance.N_neurons)

    deltas = Vector{Vector{Float64}}(undef,(N-1,))

    ldelta_w = NN_instance.loss_function_der(activations[N-1],label) .* NN_instance.act_function_der[N-1](zvecs[N-1])

    deltas[N-1] = ldelta_w

    for i in N-2:-1:1
        ldelta_w = NN_instance.act_function_der[i](zvecs[i]) .* NN_instance.layer_weights[i+1]' * ldelta_w 
        deltas[i] = ldelta_w
    end

    weight_updates = Vector{Matrix{Float64}}(undef,(N-1,))
    bias_updates = Vector{Vector{Float64}}(undef,(N-1,))

    for i in 2:N-1
        weight_updates[i] = deltas[i] .* activations[i-1]'
        bias_updates[i] = deltas[i]
    end
    weight_updates[1] = deltas[1] .* input_vector'
    bias_updates[1] = deltas[1]

    return weight_updates, bias_updates
end

function batch_update_network(input_vectors::Vector{Vector{Float64}},labels::Vector{Vector{Float64}},NN_instance::NN_network,
    learning_rate::Float64 = 1e-3)
    
    N = length(NN_instance.N_neurons)
    N_input_vecs = length(input_vectors)
    sum_d_weight, sum_bias_updates = backpropegate(input_vectors[1],labels[1],NN_instance)

    for k in 2:N_input_vecs
        temp_weight, temp_bias = backpropegate(input_vectors[k],labels[k],NN_instance)
        sum_d_weight .= sum_d_weight .+ temp_weight
        sum_bias_updates -= sum_bias_updates .+ temp_bias
    end
    new_bias = NN_instance.layer_biases .+ learning_rate .* sum_bias_updates
    new_weights = NN_instance.layer_weights .+ learning_rate .* sum_d_weight

    for index in 1:N-1
        modify_layer(index,new_weights[index],new_bias[index],NN_instance)
    end
end

function train_network(input_vectors::Vector{Vector{Float64}},labels::Vector{Vector{Float64}},NN_instance::NN_network,
    learning_rate::Float64 = 1e-3,batch_length::Int64 = 20)

    N_input = length(input_vectors)

    last_batch_length = N_input % batch_length

    number_of_full_batches = N_input ÷ batch_length

end

neuron_arr = Vector{Int64}([3,4,2,1])
funcs = Vector{Function}([sigmoid,sigmoid,sigmoid,sigmoid])
func_derivs = Vector{Function}([sigmoid_der,sigmoid_der,sigmoid_der,sigmoid_der])
test = NN_network(neuron_arr,funcs,func_derivs,binary_crossentropy,binary_crossentropy_der)

#= 
Generate data such that the label is the sum of the input vector of length 3.
=#

N_data = 1000
test_vectors = Vector{Vector{Float64}}(undef,(N_data,))
test_labels = Vector{Vector{Float64}}(undef,(N_data,))

for j in 1:N_data
    test_vectors[j] = rand(Float64,(3,))
    test_labels[j] = Vector([sum(test_vectors[j])])
end

println(test.layer_weights[1])
batch_update_network(test_vectors,test_labels,test)
println(test.layer_weights[1])