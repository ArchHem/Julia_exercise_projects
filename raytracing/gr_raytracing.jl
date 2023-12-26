using Symbolics, StaticArrays, Plots, NLsolve, ProgressBars, LinearAlgebra, Colors, Images

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

function metric_to_minkowskian_basis(metric::SMatrix{4,4,Num,16},coordinates::SVector{4,Num},cartesian_coordinates::SVector{4,Num},
    inverse_coordinates::SVector{4,Num},inverse_cartesian_coordinates::SVector{4,Num})
    #will result in insance CHR calculation times.

    jacobian_coords = Symbolics.jacobian(cartesian_coordinates,coordinates)

    println(jacobian_coords)

    local_dict = Dict([coordinates[i] => inverse_coordinates[i] for i in 1:4])

    jacobian_cartesian = Matrix{Num}(undef,(4,4))

    metric_cart_sub = Matrix{Num}(undef,(4,4))

    for j in 1:4
        for i in 1:4
            jacobian_cartesian[i,j] = substitute(jacobian_coords[i,j],local_dict)
            jacobian_cartesian[i,j] = simplify(jacobian_cartesian[i,j])
            metric_cart_sub[i,j] = substitute(metric[i,j],local_dict)
            metric_cart_sub[i,j] = simplify(metric_cart_sub[i,j])
        end
    end

    metric_cart_sub = SMatrix{4,4,Num,16}(metric_cart_sub)

    final_cartesian_metric = zeros(Num,(4,4))

    for j in 1:4
        for i in 1:4
            for u in 1:4
                for v in 1:4
                    final_cartesian_metric[u,v] = final_cartesian_metric[u,v] + jacobian_cartesian[i,u] * jacobian_cartesian[j,v] * metric_cart_sub[i,j]
                end
            end
        end
    end

    for i in 1:4
        for j in 1:4
            final_cartesian_metric[i,j] = simplify(final_cartesian_metric[i,j])
        end
    end
    println(final_cartesian_metric)
    #construct new set of coordinates
    final_cartesian_metric = SMatrix{4,4,Num,16}(final_cartesian_metric)
    #eqv. to 'usual' cartesian coordinates

    x1 = inverse_cartesian_coordinates[1]
    x2 = inverse_cartesian_coordinates[2]
    x3 = inverse_cartesian_coordinates[3]
    x4 = inverse_cartesian_coordinates[4]

    local_coords = inverse_cartesian_coordinates
    local_cart_coords = @SVector [x1, x2, x3, x4]

    #eqv to usual inverse cartesian
    @variables t_ii::Real, x_ii::Real, y_ii::Real, z_ii::Real

    local_inv_cartesian = @SVector [t_ii, x_ii, y_ii, z_ii]

    local_inv_coords = @SVector [t_ii, x_ii, y_ii, z_ii]

    return final_cartesian_metric, local_coords, local_cart_coords, local_inv_coords, local_inv_cartesian

end

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
            for i in 1:4
                temp_CH_symbols[u][i,j] = simplify(temp_CH_symbols[u][i,j])
            end
        end
    end

    for u in 1:4
        for j in 1:4
            for i in 1:j
                temp_CH_symbols[u][j,i] = temp_CH_symbols[u][i,j]
            end
        end
    end

    prelim_interm = Array{Num}(undef,(4,4,4))

    for i in 1:4
        prelim_interm[i,:,:] = temp_CH_symbols[i]
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
    alpha::Float64,coordinate_basis::Int64 = 1)::SVector{4,Float64}
    cartesian_fourveloc = metric_instance.jacobian(coord_fourpos) * coord_fourveloc
    scaler_vector = Vector{Float64}([alpha, alpha, alpha, alpha])
    scaler_vector[coordinate_basis] = 1.0
    scaler_vector = SVector{4, Float64}(scaler_vector)
    new_cart_fourveloc = scaler_vector .* cartesian_fourveloc
    new_coord_fourveloc = SVector{4,Float64}(metric_instance.inverse_jacobian(coord_fourpos) * new_cart_fourveloc)
    return new_coord_fourveloc
end

function normalize_fourveloc_bunch(metric_instance::metric_container,cart_pos::Vector{SVector{4, Float64}},
    cart_fourveloc::Vector{SVector{4, Float64}}, quant::Real = 0.0,coord_basis::Int64 = 1)
    N = length(cart_pos)
    new_coord_fourveloc_container = Vector{SVector{4, Float64}}(undef,N)

    Threads.@threads for i in ProgressBar(1:N)
        local_coord_fourpos = metric_instance.from_cartesian_to_coords(cart_pos[i])
        local_coord_fourveloc = metric_instance.inverse_jacobian(local_coord_fourpos) * cart_fourveloc[i]
        function to_solve(x::Vector{Float64})
            val = metric_inner_product(metric_instance,local_coord_fourpos,
            spatial_scale(metric_instance,local_coord_fourpos,local_coord_fourveloc,x[1],coord_basis)) - quant
            return val
        end
        
        sol = nlsolve(to_solve,[metric_instance.speed_of_light])
        rescaler = sol.zero

        new_coord_fourveloc = spatial_scale(metric_instance,local_coord_fourpos,local_coord_fourveloc,rescaler[1])
        new_coord_fourveloc_container[i] = new_coord_fourveloc
    end
    
    return new_coord_fourveloc_container
end

function STATIC_nan_to_zero(tensor::SArray{Tuple{4,4,4}, Float64, 3, 4^3})
    #used to ensure CH symbol physicallity
    new_tensor = Array{Float64}(undef,(4,4,4))
    new_tensor .= tensor
    for i in eachindex(new_tensor)
        if isnan(new_tensor[i])
            new_tensor[i] = 0.0
        end
    end
    output = SArray{Tuple{4,4,4}, Float64, 3, 4^3}(new_tensor)
    return output
end

function planar_camera_ray_generator(metric_instance::metric_container,N_x::Int64,N_y::Int64,d_pixel::Float64,
    camera_location::Vector{Float64},focal_distance::Real,x_angle::Real,y_angle::Real,z_angle::Real,norm_quant::Float64 = 0.0,
    coordinate_basis::Int64 = 1)
    
    N_rays = N_x * N_y

    y_range = collect(LinRange(-N_y*d_pixel/2,N_y*d_pixel/2,N_y))

    x_range = collect(LinRange(-N_x*d_pixel/2,N_x*d_pixel/2,N_x))

    x_mat = x_range' .* ones(N_y)

    y_mat = ones(N_x)' .* y_range

    x_vec = vcat(x_mat)
    y_vec = vcat(y_mat)

    initial_normal_vectors = [Vector{Float64}([-x_vec[i],-y_vec[i],-focal_distance]) for i in 1:N_rays]
    initial_normal_vectors = [Vector{Float64}(initial_normal_vectors[i])./sqrt(sum(initial_normal_vectors[i].^2)) for i in 1:N_rays]

    initial_position_vector = [Vector{Float64}([x_vec[i],y_vec[i],0.0]) for i in 1:N_rays]

    x_rotation_matrix = Matrix{Float64}([ [1.0 0.0 0.0]; [0.0 cos(x_angle) -sin(x_angle)]; [0.0 sin(x_angle) cos(x_angle)]])
    y_rotation_matrix = Matrix{Float64}([ [cos(y_angle) 0.0 sin(y_angle)]; [0.0 1.0 0.0]; [-sin(y_angle) 0.0 cos(y_angle)]])
    z_rotation_matrix = Matrix{Float64}([ [cos(z_angle) -sin(z_angle) 0.0]; [sin(z_angle) cos(z_angle) 0.0]; [0.0 0.0 1.0]])

    all_rot = z_rotation_matrix * y_rotation_matrix * x_rotation_matrix

    initial_normal_vectors = [all_rot * initial_normal_vectors[i] for i in 1:N_rays]
    
    initial_position_vector = [all_rot * initial_position_vector[i] for i in 1:N_rays]
    
    initial_position_vector = [vcat([camera_location[1]],initial_position_vector[i]) for i in 1:N_rays]
    initial_normal_vectors = [SVector{4, Float64}(vcat([1.0],metric_instance.speed_of_light * initial_normal_vectors[i])) for i in 1:N_rays]
    
    initial_position_vector = [SVector{4, Float64}(initial_position_vector[i] + camera_location) for i in 1:N_rays]

    initial_coord_pos = Vector{SVector{4, Float64}}(undef,N_rays)
    
    for i in 1:N_rays
        initial_coord_pos[i] = SVector{4,Float64}(metric_instance.from_cartesian_to_coords(initial_position_vector[i]) )
    end
    initial_coord_velocs = normalize_fourveloc_bunch(metric_instance,initial_position_vector,initial_normal_vectors,norm_quant,coordinate_basis)
    return initial_coord_pos, initial_coord_velocs

end

function calculate_fouracc(metric_instance::metric_container,coord_fourpos::Vector{SVector{4, Float64}},
    coord_fourveloc::Vector{SVector{4, Float64}},coordinate_basis::Int64 = 1)
    N_rays = length(coord_fourpos)
    coord_four_acceleration = Vector{SVector{4, Float64}}(undef,N_rays)
    #hopefully matches einsum-like performance
    Threads.@threads for n in 1:N_rays
        local_acc = zeros(Float64,4)
        CHR_symbol_init = metric_instance.numeric_CH_symbol(coord_fourpos[n])
        CHR_symbol = STATIC_nan_to_zero(CHR_symbol_init)
        local_veloc = coord_fourveloc[n]
        for u in 1:4
            local_acc[u] = -(local_veloc'CHR_symbol[u,:,:] * local_veloc) + (local_veloc'CHR_symbol[coordinate_basis,:,:] * local_veloc * local_veloc[u])
        end
        coord_four_acceleration[n] = SVector{4,Float64}(local_acc)
    end 
    return coord_fourveloc, coord_four_acceleration
end

function SCH_termination_cause(coord_fourpos::Vector{SVector{4, Float64}}, coord_fourveloc::Vector{SVector{4, Float64}},
    current_indices::Vector{Int64})
    N_current = length(current_indices)
    global_indices_to_del = Vector{Int64}()
    local_indices_to_del = Vector{Int64}()
    
    for i in 1:N_current
        if coord_fourpos[i][2] < 2.025 || coord_fourpos[i][2] > 30.0
            push!(global_indices_to_del,current_indices[i])
            push!(local_indices_to_del,i)
        end
    end
    return global_indices_to_del, local_indices_to_del

end

function SCH_d0_scaler(coord_fourpos::Vector{SVector{4, Float64}}, coord_fourveloc::Vector{SVector{4, Float64}},
    d0_inner::Float64 = -0.025,d0_outer::Float64 = -0.05,zone_separator::Float64 = 15.0)

    N_current = length(coord_fourpos)

    d0 = ones(N_current) * d0_outer

    for i in 1:N_current
        if coord_fourpos[i][2] < zone_separator
            d0[i] = d0_inner
        end
    end

    return d0
end


function integrate_geodesics_no_tracking_RK4(metric_instance::metric_container,coord_fourpos::Vector{SVector{4, Float64}},
    coord_fourveloc::Vector{SVector{4, Float64}},termination::Function,d0_scaler::Function,
    coordinate_basis::Int64 = 1,N_timesteps::Int64 = 2000)

    N_init = length(coord_fourveloc)
    index_tracker = Vector{Int64}(collect(1:N_init))
    initial_fourpos = copy(coord_fourpos)
    initial_fourveloc = copy(coord_fourveloc)

    final_fourpos = Vector{SVector{4, Float64}}(undef, N_init)
    final_fourvelocity = Vector{SVector{4, Float64}}(undef, N_init)

    #can be used to render eg accreration disk
    #auxillary_color_data = Vector{Vector{Float64}}([zeros(3) for i in 1:N_init])

    for t in ProgressBar(1:N_timesteps)
        
        if length(index_tracker) == 0
            println("All terms terminated at timestep " * string(t))
            break
        end

        local_d0 = d0_scaler(coord_fourpos,coord_fourveloc)

        d1_fourpos, d1_fourveloc = calculate_fouracc(metric_instance,coord_fourpos,coord_fourveloc,coordinate_basis)
        
        d2_fourpos, d2_fourveloc = calculate_fouracc(metric_instance,
        coord_fourpos .+ 0.5 .* local_d0 .* d1_fourpos, coord_fourveloc .+ 0.5 .* local_d0 .* d1_fourveloc, coordinate_basis)

        d3_fourpos, d3_fourveloc = calculate_fouracc(metric_instance,
        coord_fourpos .+ 0.5 .* local_d0 .* d2_fourpos, coord_fourveloc .+ 0.5 .* local_d0 .* d2_fourveloc, coordinate_basis)

        d4_fourpos, d4_fourveloc = calculate_fouracc(metric_instance,
        coord_fourpos .+  local_d0 .* d3_fourpos, coord_fourveloc .+ local_d0 .* d3_fourveloc, coordinate_basis)

        coord_fourpos = @. coord_fourpos + local_d0/6 * (d1_fourpos + 2 * d2_fourpos + 2 * d3_fourpos + d4_fourpos)
        coord_fourveloc = @. coord_fourveloc + local_d0/6 * (d1_fourveloc + 2 * d2_fourveloc + 2 * d3_fourveloc + d4_fourveloc)

        global_del, local_del = termination(coord_fourpos,coord_fourveloc,index_tracker)
        
        if length(global_del) > 0
            
            final_fourpos[global_del] = coord_fourpos[local_del]
            final_fourvelocity[global_del] = coord_fourveloc[local_del]
            index_tracker = deleteat!(index_tracker,local_del)
            coord_fourpos = deleteat!(coord_fourpos,local_del)
            coord_fourveloc = deleteat!(coord_fourveloc,local_del)
        end 

    end

    final_fourpos[index_tracker] = coord_fourpos
    final_fourvelocity[index_tracker] = coord_fourveloc

    println(string(length(index_tracker)) * " rays remain underminated.")

    return initial_fourpos, initial_fourveloc, final_fourpos, final_fourvelocity
end

function SCH_colorer(final_fourvectors::Vector{SVector{4, Float64}},final_fourvelocs::Vector{SVector{4, Float64}},image::Matrix{RGBA{N0f8}})
    for i in eachindex(image)
        if final_fourvectors[i][2] < 2.025
            
            image[i] = RGBA{N0f8}(0.0,0.0,0.0,1.0)
        end
    end
    return image
end

function standard_CS_renderer(image_path::String, metric_instance::metric_container,coord_fourpos::Vector{SVector{4, Float64}},
    coord_fourveloc::Vector{SVector{4, Float64}},termination::Function,d0_scaler::Function, N_x_cam::Int64, N_y_cam::Int64,custom_colorer::Function,
    coordinate_basis::Int64 = 1,N_timesteps::Int64 = 2000)

    celestial_sphere = load(image_path)

    Ny, Nx = size(celestial_sphere)

    output_image = Matrix{eltype(celestial_sphere)}(undef,(N_y_cam,N_x_cam))

    starting_fourpos, starting_fourveloc, final_fourpos, final_fourveloc = integrate_geodesics_no_tracking_RK4(metric_instance,
    coord_fourpos,coord_fourveloc,termination,d0_scaler,coordinate_basis,N_timesteps)

    N_rays = length(final_fourveloc)
    minkowsi_coords = Vector{SVector{4, Float64}}(undef,N_rays)
    minkowsi_velocity = Vector{SVector{4, Float64}}(undef,N_rays)


    for i in 1:N_rays
        minkowsi_coords[i] = metric_instance.from_coords_to_cartesian(final_fourpos[i])
        minkowsi_velocity[i] = metric_instance.jacobian(final_fourpos[i]) * final_fourveloc[i]
    end


    #map minkowskian velocity into celestial sphere

    quasi_r = [sqrt(minkowsi_velocity[i][4]^2 + minkowsi_velocity[i][3]^2 + minkowsi_velocity[i][2]^2) for i in 1:N_rays]

    quasi_theta = [acos(minkowsi_velocity[i][4]/quasi_r[i]) for i in 1:N_rays]
    quasi_phi = [atan(minkowsi_velocity[i][3],minkowsi_velocity[i][2]) for i in 1:N_rays]
    quasi_phi = @. (quasi_phi + 2pi) % (2pi)

    quasi_theta = reshape(quasi_theta, (N_y_cam,N_x_cam))
    quasi_phi = reshape(quasi_phi, (N_y_cam,N_x_cam))

    for j in 1:N_x_cam
        for i in 1:N_y_cam
            y_index = ceil(Int64,quasi_theta[i,j]*Ny/(pi) ) 
            x_index = ceil(Int64,quasi_phi[i,j]*Nx/(2pi) ) 
            
            output_image[i,j] = celestial_sphere[y_index, x_index]
        end
    end
    
    
    output_image = custom_colorer(final_fourpos,final_fourveloc,output_image)

    return output_image

end
test_container = metric_container(sch_metric_representation,coords,cartesian_coords,inverse_coords,inverse_cartesian_coords,1.0)

Nx = 100
Ny = 100
fourvec0, fourveloc0 = planar_camera_ray_generator(test_container,Nx,Ny,0.075,Vector([0.0,0.0,5.0,0.0]),1.0,+pi/2,0.0,0.0)

test1, test2 = calculate_fouracc(test_container,fourvec0,fourveloc0)


output_image = standard_CS_renderer("raytracing/celestial_spheres/QUASI_CS.png",test_container,fourvec0,fourveloc0,SCH_termination_cause,SCH_d0_scaler,
Nx,Ny,SCH_colorer,1,4000)
save("raytracing/renders/test_04.png",output_image)
println("test")