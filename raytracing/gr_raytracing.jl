using Symbolics, StaticArrays

@variables x1::Real, x2::Real, x3::Real, x4::Real
@variables t_i::Real, x_i::Real, y_i::Real, z_i::Real

c = 1.0
M = 1.0
G = 1.0
r_s = 2.0 * G * M / c^2

g_00 = -(1.0 - r_s / x2) * c^2 
g_11 =  (1.0 - r_s / x2)^(-1.0)
g_22 = x2^2 * sin(x4)^2
g_33 = x2^2 

sch_metric_representation = @SMatrix [
    g_00 0.0 0.0 0.0;
    0.0 g_11 0.0 0.0;
    0.0 0.0 g_22 0.0;
    0.0 0.0 0.0 g_33
]

t = x1
x = x2 * sin(x4) * cos(x3)
y = x2 * sin(x4) * sin(x3)
z = x2 * cos(x4)

x1_i = t_i
x2_i = sqrt(x_i^2 + y_i^2 + z_i^2)
x3_i = atan(y_i, x_i)
x4_i = acos(z_i/x2_i)

coords = SVector(x1, x2, x3, x4)
cartesian_coords = SVector(t,x,y,z)

inverse_cartesian_coords = SVector(t_i,x_i,y_i,z_i)
inverse_coords = SVector(x1_i, x2_i, x3_i, x4_i)

#create a callable metric function using Symbolics.jl package; output can be mutative or allocating type
function numeric_matrix_generator(matrix::StaticArray,coordinates::SVector{4,Num})
    new_functions = build_function(matrix, coordinates)
    allocating_matrix = eval(new_functions[1])
    mutating_matrix= eval(new_functions[2])
    return allocating_matrix, mutating_matrix
end

function generate_jacobians(cartesian_coords::SVector{4,Num},coordinates::SVector{4,Num})
    jacobian_temp = Symbolics.jacobian(cartesian_coords,coordinates,simplify=true)
    jacobian_temp_inverse = inv(jacobian_temp)

    temp_callable_jac = build_function(jacobian_temp,coordinates)
    temp_callable_inv_jac = build_function(jacobian_temp_inverse,coordinates)
    #not interested in mutability.
    final_jac = eval(temp_callable_jac[1])
    final_inv_jac = eval(temp_callable_inv_jac[1])
    return final_jac, final_inv_jac
end

function generate_coordinate_transforms(coords::SVector{4,Num},derived_cartesian::SVector{4,Num},
    cartesian::SVector{4,Num},derived_coords::SVector{4,Num})
    
    #build to_cartesian function first
    from_coords_to_cartesian = eval(build_function(derived_cartesian,coords)[1])
    #now the reverse
    #TODO? try to implement automatic inversion
    from_cartesian_to_coords = eval(build_function(derived_coords,cartesian)[1])
    return from_coords_to_cartesian, from_cartesian_to_coords
end

function generate_christoffel_symbol(metric::SMatrix{4,4,Num,16},coordinates::SVector{4,Num})
    differential_operators = @SVector [Differential(coordinates[i]) for i in 1:4]
    inverse_metric = inv(metric)

    temp_derivs = [Matrix{Num}(undef,(4,4)) for i in 1:4]
    #differentiation of Array-like objects is not supported yet.
    for k in 1:4
        for j in 1:4
            for i in 1:4
                temp_derivs[k][i,j] = expand_derivatives(differential_operators[k](metric[i,j]))
            end
        end
    end
    #now we create the Christoffel symbol holders.
    temp_CH_symbols = [zeros(Num,4,4) for i in 1:4]
    
    for u in 1:4
        for m in 1:4
            for j in 1:4
                for i in 1:j
                    temp_CH_symbols[u][i,j] = temp_CH_symbols[u][i,j] + 0.5 * inverse_metric[u,m] * (
                    temp_derivs[j][m,i] + temp_derivs[i][m,j] - temp_derivs[m][i,j]
                    )
                end
            end
        end
    end

    #reflect values on diagonal, using lower index symetry

    for u in 1:4
        for j in 1:4
            for i in 1:j
                temp_CH_symbols[u][j,i] = temp_CH_symbols[u][i,j]
            end
        end
    end
    
    interm = [simplify(temp_CH_symbols[i]) for i in 1:4]

    prelim_interm = Array{Num}(undef,(4,4,4))

    for i in 1:4
        prelim_interm[i,:,:] = interm[i]
    end
    
    simplified_CH_symbols = SArray{Tuple{4,4,4}, Num, 3, 4^3}(prelim_interm)
    #We want everything static for performance and non-mutability
    return simplified_CH_symbols    
end

CH_symbol = generate_christoffel_symbol(sch_metric_representation,coords)
println(typeof(CH_symbol))
CH_func = numeric_matrix_generator(CH_symbol,coords)[1]
CH_func2 = numeric_matrix_generator(CH_symbol,coords)[1]
println(typeof(CH_func),typeof(CH_func2))

struct metric_container{T1<:Function,T2<:Function,T3<:Function,T4<:Function,T5<:Function}
    #is this too OOP-like?
    metric::SMatrix{4,4,Num,16}
    coordinates::SVector{4,Num}
    cartesian_coordinates::SVector{4,Num}
    inverse_coordinates::SVector{4,Num}
    inverse_cartesian_coordinates::SVector{4,Num}
    #non-inputs
    CH_symbols::SArray{Tuple{4, 4, 4}, Num, 3, 64}
    numeric_CH_symbol::T1
    jacobian::T2
    inverse_jacobian::T3
    from_coords_to_cartesian::T4
    from_cartesian_to_coords::T5
    function metric_container(metric::SMatrix{4,4,Num,16},
        coordinates::SVector{4,Num},
        cartesian_coordinates::SVector{4,Num},
        inverse_coordinates::SVector{4,Num},
        inverse_cartesian_coordinates::SVector{4,Num})
    
    CH_symbols = generate_christoffel_symbol(metric,coordinates)
    numeric_CH_symbol = numeric_matrix_generator(CH_symbols,coordinates)[1]
    jacobian, inverse_jacobian = generate_jacobians(cartesian_coordinates,coordinates)
    from_coords_to_cartesian, from_cartesian_to_coords = generate_coordinate_transforms(coordinates,cartesian_coordinates,
                                                                    inverse_coordinates,inverse_cartesian_coordinates)
    
    new{typeof(numeric_CH_symbol),typeof(jacobian),typeof(inverse_jacobian),typeof(from_coords_to_cartesian),typeof(from_cartesian_to_coords)}(
    metric, coordinates, cartesian_coordinates, inverse_coordinates,inverse_cartesian_coordinates,
    CH_symbols,numeric_CH_symbol,jacobian,inverse_jacobian,from_coords_to_cartesian,from_cartesian_to_coords)
    end
end

test = metric_container(sch_metric_representation,coords,cartesian_coords,inverse_coords,inverse_cartesian_coords)

println(typeof(test.CH_symbols[1,:,:]))