function sigmoid(x::Vector{Float64})
    quant =  1.0 ./ (1.0 .+ exp.(.-x))
    return quant
end

function binary_crossentropy(ypredict::Vector{Float64},
    ylabel::Vector{Float64})
    quant = -ypredict .* log.(ylabel) - (1.0 .- ylabel) .* log.(1.0 .- ypredict)
    return quant[1]
end

struct NN_network
    N_neurons::Vector{Int64}
    layer_weights::Vector{Matrix{Float64}}
    layer_biases::Vector{Vector{Float64}}
    act_functions::Vector{Function}
    loss_function::Function
    function NN_network(N_neurons::Vector{Int64},act_functions::Vector{Function},
        loss_function::Function)
        layer_weights = Vector{Matrix{Float64}}(undef,length(N_neurons)-1)
        layer_biases = Vector{Vector{Float64}}(undef,length(N_neurons)-1)
        for i in 1:length(N_neurons)-1
            layer_weights[i] = rand(Float64,(N_neurons[i+1],N_neurons[i]))
            layer_biases[i] = rand(Float64,(N_neurons[i+1],))
        end
        new(N_neurons,layer_weights,layer_biases,act_functions,loss_function)
    end
end 

function apply_layer(input_vector::Vector{Float64},NN_instance::NN_network,index::Int64)
    l_weights = NN_instance.layer_weights[index,]
    l_bias = NN_instance.layer_biases[index,]
    l_afunc = NN_instance.act_functions[index,]

    output_vector = l_afunc(l_weights * input_vector + l_bias)

    return output_vector
end

function propegate_thr_network(input_vector::Vector{Float64},NN_instance::NN_network)
    number_of_layers = length(NN_instance.N_neurons)
    output_vector = input_vector
    for indices in 1:number_of_layers-1
        output_vector = apply_layer(output_vector,NN_instance,indices)
    end
    return output_vector
end

function evaluate_loss(input_vector::Vector{Float64},label::Vector{Float64},NN_instance::NN_network)
    output = propegate_thr_network(input_vector,NN_instance)
    loss = NN_instance.loss_function(output,label)
    return loss
end

function evaluate_batch_avg_loss(input_vectors::Matrix{Float64},
    labels::Vector{Float64},NN_instance::NN_network)

    if length(labels) != size(input_vectors)[1]
        throw(ArgumentError("Shape mismatch in predicted and known labels."))
    end 
    N = length(labels)
    losses = Vector{Float64}(undef,(N,))

    for j in 1:N
        losses[j] = evaluate_loss(input_vectors[j,:],labels[j,:],NN_instance)
    end
    
    avg = sum(losses)/N
    return avg
end

function modify_layer(layer_index::Int64,new_weights::Matrix{Float64},new_biases::Vector{Float64},NN_instance::NN_network)
    NN_instance.layer_weights[layer_index] .= new_weights
    NN_instance.layer_biases[layer_index] .= new_biases
end



neuron_arr = Vector{Int64}([3,4,2,1])
funcs = Vector{Function}([sigmoid,sigmoid,sigmoid,sigmoid])
test = NN_network(neuron_arr,funcs,binary_crossentropy)
test_vec = Vector{Float64}([1.0,2.0,3.0])
test_batch = rand(Float64,(10,3))
test_labels = rand(Float64,(10,))

println(apply_layer(test_vec,test,1))
println(propegate_thr_network(test_vec,test))
println(evaluate_batch_avg_loss(test_batch,test_labels,test))

println(test.layer_weights[3])
println(test.layer_biases[3])
modify_layer(3,Matrix{Float64}(undef,(1,2)),Vector{Float64}(undef,(1,)),test)
println(test.layer_weights[3])
println(test.layer_biases[3])
