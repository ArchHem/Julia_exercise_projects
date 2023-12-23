using Symbolics, StaticArrays, Plots, NLsolve

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
    derived_coords::SVector{4,Num},cartesian::SVector{4,Num})
    
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

struct metric_container{T1<:Function,T2<:Function,T3<:Function,T4<:Function,T5<:Function,T6<:Function}
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
    numeric_metric::T6
    speed_of_light::Float64

    function metric_container(metric::SMatrix{4,4,Num,16},
        coordinates::SVector{4,Num},
        cartesian_coordinates::SVector{4,Num},
        inverse_coordinates::SVector{4,Num},
        inverse_cartesian_coordinates::SVector{4,Num},speed_of_light::Float64)
    
    CH_symbols = generate_christoffel_symbol(metric,coordinates)
    numeric_CH_symbol = numeric_matrix_generator(CH_symbols,coordinates)[1]
    numeric_metric = numeric_matrix_generator(metric,coordinates)[1]
    jacobian, inverse_jacobian = generate_jacobians(cartesian_coordinates,coordinates)
    from_coords_to_cartesian, from_cartesian_to_coords = generate_coordinate_transforms(coordinates,cartesian_coordinates,
                                                                    inverse_coordinates,inverse_cartesian_coordinates)
    
    new{typeof(numeric_CH_symbol),typeof(jacobian),typeof(inverse_jacobian),typeof(from_coords_to_cartesian),
    typeof(from_cartesian_to_coords),typeof(numeric_metric)
    }(
    metric, coordinates, cartesian_coordinates, inverse_coordinates,inverse_cartesian_coordinates,
    CH_symbols,numeric_CH_symbol,jacobian,inverse_jacobian,from_coords_to_cartesian,from_cartesian_to_coords,
    numeric_metric,speed_of_light)
    end
end

function metric_inner_product(metric_instance::metric_container,coord_fourpos::SVector{4, Float64},coord_fourveloc::SVector{4, Float64})::Float64
    metric_value = metric_instance.numeric_metric(coord_fourpos)
    inner_product = sum(coord_fourveloc .* (metric_value * coord_fourveloc))
    return inner_product
end

function spatial_scale(metric_instance::metric_container,coord_fourpos::SVector{4, Float64},coord_fourveloc::SVector{4, Float64},
    alpha::Float64)::SVector{4,Float64}
    cartesian_fourveloc = metric_instance.jacobian(coord_fourpos) * coord_fourveloc
    scaler_vector = SVector{4, Float64}([1.0, alpha, alpha, alpha])
    new_cart_fourveloc = scaler_vector .* cartesian_fourveloc
    new_coord_fourveloc = SVector{4,Float64}(metric_instance.inverse_jacobian(coord_fourpos) * new_cart_fourveloc)
    return new_coord_fourveloc
end

function normalize_fourveloc_bunch(metric_instance::metric_container,cart_pos::Vector{SVector{4, Float64}},
    cart_fourveloc::Vector{SVector{4, Float64}}, quant::Float64 = 0.0)
    N = length(cart_pos)
    new_coord_fourveloc_container = Vector{SVector{4, Float64}}(undef,N)

    for i in 1:N
        local_coord_fourpos = metric_instance.from_cartesian_to_coords(cart_pos[i])
        local_coord_fourveloc = metric_instance.inverse_jacobian(local_coord_fourpos) * cart_fourveloc[i]
        function to_solve(x::Vector{Float64})
            val = metric_inner_product(metric_instance,local_coord_fourpos,
            spatial_scale(metric_instance,local_coord_fourpos,local_coord_fourveloc,x[1])) - quant
            return val
        end
        
        sol = nlsolve(to_solve,[metric_instance.speed_of_light])
        rescaler = sol.zero
        new_coord_fourveloc = spatial_scale(metric_instance,local_coord_fourpos,local_coord_fourveloc,rescaler[1])
        new_coord_fourveloc_container[i] = new_coord_fourveloc
    end
    
    return new_coord_fourveloc_container
end

function planar_camera_ray_generator(metric_instance::metric_container,N_x::Int64,N_y::Int64,d_pixel::Float64,
    camera_location::Vector{Float64},focal_distance::Real,x_angle::Real,y_angle::Real,z_angle::Real,
    norm_quant::Real)
    
    N_rays = N_x * N_y

    y_range = collect(LinRange(-N_y*d_pixel/2,N_y*d_pixel/2,N_y))

    x_range = collect(LinRange(-N_x*d_pixel/2,N_x*d_pixel/2,N_x))

    x_mat = x_range' .* ones(N_y)

    y_mat = ones(N_x)' .* y_range

    x_vec = vcat(x_mat)
    y_vec = vcat(y_mat)

    initiaL_normal_vectors = [Vector{Float64}([x_vec[i],y_vec[i],focal_distance]) for i in 1:N_rays]
    initiaL_normal_vectors = [Vector{Float64}(initiaL_normal_vectors[i])./sqrt(sum(initiaL_normal_vectors[i].^2)) for i in 1:N_rays]

    initial_position_vector = [Vector{Float64}([x_vec[i],y_vec[i],0.0]) for i in 1:N_rays]

    x_rotation_matrix = Matrix{Float64}([ [1.0 0.0 0.0]; [0.0 cos(x_angle) -sin(x_angle)]; [0.0 sin(x_angle) cos(x_angle)]])
    y_rotation_matrix = Matrix{Float64}([ [cos(y_angle) 0.0 sin(y_angle)]; [0.0 1.0 0.0]; [-sin(y_angle) 0.0 cos(y_angle)]])
    z_rotation_matrix = Matrix{Float64}([ [cos(z_angle) -sin(z_angle) 0.0]; [sin(z_angle) cos(z_angle) 0.0]; [0.0 0.0 1.0]])

    all_rot = z_rotation_matrix * y_rotation_matrix * x_rotation_matrix

    initiaL_normal_vectors = [all_rot * initiaL_normal_vectors[i] for i in 1:N_rays]
    
    initial_position_vector = [all_rot * initial_position_vector[i] for i in 1:N_rays]
    
    initial_position_vector = [vcat([0.0],initial_position_vector[i]) for i in 1:N_rays]
    initiaL_normal_vectors = [SVector{4, Float64}(vcat([1.0],metric_instance.speed_of_light * initiaL_normal_vectors[i])) for i in 1:N_rays]
    
    initial_position_vector = [SVector{4, Float64}(initial_position_vector[i] + camera_location) for i in 1:N_rays]

    initial_coord_pos = Vector{SVector{4, Float64}}(undef,N_rays)
    for i in 1:N_rays
        initial_coord_pos[i] = SVector{4,Float64}(metric_instance.from_cartesian_to_coords(initial_position_vector[i]) )
    end
    initial_coord_velocs = normalize_fourveloc_bunch(metric_instance,initial_position_vector,initiaL_normal_vectors)
    return initial_coord_pos, initial_coord_velocs

end

function calculate_fouracc(metric_instance::metric_container,coord_fourpos::Vector{SVector{4, Float64}},
    coord_fourveloc::Vector{SVector{4, Float64}})
    Number_of_rays = length(coord_fourveloc)
    coord_four_acceleration = Vector{SVector{4, Float64}}(undef,Number_of_rays)

    Threads.@threads for n in 1:Number_of_rays
        
    end 
end

test_container = metric_container(sch_metric_representation,coords,cartesian_coords,inverse_coords,inverse_cartesian_coords,1.0)

fourvec0, fourveloc0 = planar_camera_ray_generator(test_container,50,50,0.01,Vector([0.0,0.0,2.1,0.0]),5,-pi/2,0,0,1)


