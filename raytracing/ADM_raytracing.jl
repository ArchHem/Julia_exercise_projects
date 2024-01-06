
module ADM_raytracing
using Symbolics, StaticArrays, ProgressBars, LinearAlgebra, Colors, Images

function permutation_sign(perm::AbstractVector{Int64})
    #only works for unique permutations
    L = length(perm)
    crosses = 0
    for i = 1:L
        for j = i+1 : L
            crosses += perm[j] < perm[i]
        end
    end
    return iseven(crosses) ? 1 : -1    
end

function levi_civita_generator()
    outp = zeros(Int64,4,4,4,4)
    for d in 1:4
        
        for c in 1:4
            
            for b in 1:4
                
                for a in 1:4

                    number_uniq = length(unique([a,b,c,d]))
                    
                    if number_uniq == 4
                        
                        outp[a,b,c,d] = permutation_sign([a,b,c,d])
                    end

                end
            end
        end
    end
    return outp
end

function numeric_array_generator(matrix,coordinates)
    new_functions = build_function(matrix, coordinates)
    
    allocating_matrix = @inline eval(new_functions[1])
    mutating_matrix = @inline eval(new_functions[2])
    
    return allocating_matrix, mutating_matrix
end

function generate_christoffel_symbol(metric::SMatrix{4,4,Num,16},coordinates::SVector{4,Num})

    differential_operators = @SVector [Differential(coordinates[i]) for i in 1:4]
    inverse_metric = inv(metric)

    temp_derivs = [Matrix{Num}(undef,(4,4)) for i in 1:4]
    
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
                temp_CH_symbols[u][i,j] = temp_CH_symbols[u][i,j]
            end
        end
    end

    println("Beginning simplification of Christoffel Symbols.")

    for u in ProgressBar(1:4)
        for j in 1:4
            for i in 1:j
                temp_CH_symbols[u][j,i] = simplify(temp_CH_symbols[u][i,j])
            end
        end
    end

    prelim_interm = Array{Num}(undef,(4,4,4))

    for i in 1:4
        prelim_interm[i,:,:] = temp_CH_symbols[i]
    end
    
    simplified_CH_symbols = SArray{Tuple{4,4,4}, Num, 3, 4^3}(prelim_interm)
    
    return simplified_CH_symbols    
end

struct ADM_metric_container{TNumMetric<:Function,TInverseNumMetric<:Function,TNumAcceleration<:Function,TNumericU0<:Function}
    symbolic_metric::SMatrix{4,4,Num,16}
    inverse_metric::SMatrix{4,4,Num,16}
    coordinates::SVector{4,Num}
    Christoffel_symbols::SArray{Tuple{4, 4, 4}, Num, 3, 64}
    bound_epsilon::Int64

    #symbolic functions
    alpha::Num
    beta::SVector{3,Num}
    gamma::SMatrix{3,3,Num,9}

    #numeric functions
    numeric_metric::TNumMetric
    numeric_inverse_metric::TInverseNumMetric
    numeric_acceleration::TNumAcceleration
    numeric_u0::TNumericU0


    

    function ADM_metric_container(metric_representation::SMatrix{4,4,Num,16},coordinates::SVector{4,Num},null_geodesic::Bool = true)

        chr_symbols = generate_christoffel_symbol(metric_representation,coordinates)
        inverse_sym_metric = inv(metric_representation)

        if null_geodesic
            epsilon= 0.0
        else
            epsilon = 1.0
        end

        gamma = metric_representation[2:4,2:4]
        beta = metric_representation[1,2:4]
        gamma_up = inv(gamma)
        beta_up = simplify(gamma_up * beta)
        alpha = simplify(sqrt(beta'beta_up - metric_representation[1,1]))

        # https://iopscience.iop.org/article/10.3847/1538-4365/aac9ca/pdf?fbclid=IwAR0pORzJb6EvCVdTIWo32F6wxhdd3_eQE_-x8afe94Y8dY_2IH_NuNcPiD0

        acceleration_vector =  zeros(Num,8)

        acceleration_vector[1] = 1.0
        acceleration_vector[5] = 0.0
        
        @variables u0::Real, u1::Real, u2::Real, u3::Real

        all_lower_u = [u0, u1, u2, u3]

        lower_spatial_u = [u1, u2, u3]

        implicit_u0 = sqrt(lower_spatial_u'gamma_up*lower_spatial_u + epsilon)/alpha

        u0_generator_output = MVector{8,Num}([([coordinates,[implicit_u0,u1,u2,u3]]...)...])
        
        dx_spatial = (gamma_up * lower_spatial_u) ./ u0 .- beta_up

        dx_spatial .= simplify.(dx_spatial)

        acceleration_vector[2:4] = dx_spatial

        differential_operators = [Differential(coordinates[i]) for i in 2:4]

        du_spatial = zeros(Num,(3,))

        for i in 1:3
            alpha_deriv = expand_derivatives(differential_operators[i](alpha))
            beta_up_deriv = zeros(Num,(3,))
            gamma_up_deriv = zeros(Num,(3,3))

            for k in eachindex(beta_up_deriv)
                beta_up_deriv[k] = expand_derivatives(differential_operators[i](beta_up[k]))
            end
            for k in eachindex(gamma_up_deriv)
                gamma_up_deriv[k] = expand_derivatives(differential_operators[i](gamma_up[k]))
            end
            
            du_spatial[i] = -alpha * u0 * alpha_deriv + lower_spatial_u'beta_up_deriv - 1/(2 * u0) * (lower_spatial_u'gamma_up_deriv*lower_spatial_u)
        end

        du_spatial .= simplify.(du_spatial)

        acceleration_vector[6:8] = du_spatial

        inputs = zeros(Num,8)
        for i in 1:4
            inputs[i] = coordinates[i]
            inputs[4+i] = all_lower_u[i]
        end
        

        numeric_metric = numeric_array_generator(metric_representation,coordinates)[1]

        inverse_numeric_metric = numeric_array_generator(inverse_sym_metric,coordinates)[1]

        acceleration_function = numeric_array_generator(acceleration_vector,inputs)[1]
        
        numeric_u0 = numeric_array_generator(u0_generator_output,inputs)[2]

        T0 = typeof(numeric_metric)
        T1 = typeof(inverse_numeric_metric)
        T2 = typeof(acceleration_function)
        T3 = typeof(numeric_u0)

        new{T0,T1,T2,T3}(metric_representation,inverse_sym_metric,coordinates,chr_symbols,epsilon,
        alpha,beta,gamma,
        numeric_metric,inverse_numeric_metric,acceleration_function,numeric_u0)
    end
end

function integrate_ADM_geodesics_RK4_tracked(metric_binder::ADM_metric_container,initial_allvector::Vector{MVector{8,Float64}}, 
    number_of_steps::Int64, dt_scaler::Function)

    function multi_acc(allvectors::Vector{MVector{8,Float64}})
        N_rays = length(allvectors)
        outp = Vector{MVector{8,Float64}}(undef,(N_rays,))
        Threads.@threads for i in 1:N_rays
            metric_binder.numeric_u0(allvectors[i],allvectors[i])
            temp = metric_binder.numeric_acceleration(allvectors[i])
            for j in 1:8
                @inbounds temp[j] = ifelse((isfinite(temp[j])), temp[j], 0.0)
            end
            outp[i] = temp
        end
        
        return outp
    end

    data_storage = [copy(initial_allvector) for i in 1:number_of_steps]
    

    allvector = copy(initial_allvector)
    

    for t in ProgressBar(1:number_of_steps)

        data_storage[t] = copy(allvector)

        d0 = dt_scaler(metric_binder,allvector)

        d1_allvec = multi_acc(allvector)

        d2_allvec = multi_acc(allvector .+ 0.5 .* d0 .* d1_allvec)

        d3_allvec = multi_acc(allvector .+ 0.5 .* d0 .* d2_allvec)

        d4_allvec = multi_acc(allvector .+ d0 .* d3_allvec)

        @. allvector += d0/6 * (d1_allvec + 2 * d2_allvec + 2 * d3_allvec + d4_allvec)

    end

    return data_storage

end

function integrate_ADM_geodesics_RK4(metric_binder::ADM_metric_container,initial_allvector::Vector{MVector{8,Float64}}, 
    number_of_steps::Int64, dt_scaler::Function, terminator_function::Function)

    function multi_acc(allvectors::Vector{MVector{8,Float64}})
        N_rays = length(allvectors)
        outp = Vector{MVector{8,Float64}}(undef,(N_rays,))
        Threads.@threads for i in 1:N_rays
            metric_binder.numeric_u0(allvectors[i],allvectors[i])
            temp = metric_binder.numeric_acceleration(allvectors[i])
            for j in 1:8
                @inbounds temp[j] = ifelse((isfinite(temp[j])), temp[j], 0.0)
            end
            outp[i] = temp
        end
        
        return outp
    end

    allvector = copy(initial_allvector)
    N_init = length(allvector)
    index_tracker = Vector{Int64}(collect(1:N_init))

    final_allvector = Vector{MVector{8,Float64}}(undef,N_init)

    for t in ProgressBar(1:number_of_steps)
        
        if length(index_tracker) == 0
            println("All terms terminated at timestep " * string(t))
            break
        end

        d0 = dt_scaler(metric_binder,allvector)

        d1_allvec = multi_acc(allvector)

        d2_allvec = multi_acc(allvector .+ 0.5 .* d0 .* d1_allvec)

        d3_allvec = multi_acc(allvector .+ 0.5 .* d0 .* d2_allvec)

        d4_allvec = multi_acc(allvector .+ d0 .* d3_allvec)

        @. allvector += d0/6 * (d1_allvec + 2 * d2_allvec + 2 * d3_allvec + d4_allvec)

        

        global_del, local_del = terminator_function(metric_binder,allvector, index_tracker)

        if length(global_del) > 0
            final_allvector[global_del] = allvector[local_del]
            index_tracker = deleteat!(index_tracker,local_del)
            allvector = deleteat!(allvector,local_del)
            
        end 

    end

    final_allvector[index_tracker] = allvector
    

    println(string(length(index_tracker)) * " rays remain underminated.")

    return final_allvector

end

function camera_rays_generator(metric_binder::ADM_metric_container,
    initial_fourpos::MVector{4,Float64},initial_fourvelocity::MVector{4,Float64},
    camera_front_vector::MVector{4,Float64},camera_up_vector::MVector{4,Float64},
    angular_pixellation::Float64 = 0.001,N_x::Int64 = 400,N_y::Int64 = 200)

    local_metric = metric_binder.numeric_metric(initial_fourpos)
    local_inverse_metric = metric_binder.numeric_inverse_metric(initial_fourpos)
    

    levi = levi_civita_generator()
    l_eps = metric_binder.bound_epsilon

    metric_determinant = det(local_metric)

    initial_norm = initial_fourvelocity'local_metric*initial_fourvelocity

    if initial_norm > 0.0
        throw(ArgumentError("Spacelike four-velocity given for the camera."))
    end
    if (camera_front_vector[1] == camera_up_vector[1] == 0.0) != true
        throw((ArgumentError("Camera alignment vectors must have a zero temporal part!")))
    end

    normalizing_quant = sqrt(-1.0/initial_norm)

    initial_fourvelocity = normalizing_quant .* initial_fourvelocity

    e0 = copy(initial_fourvelocity)

    #since v1, v2 is going to be a spacelike vector....
    
    v1 = camera_front_vector + (e0'local_metric*camera_front_vector) .* e0

    e1 = v1 ./ sqrt(v1'local_metric*v1)

    #whener we are projecting unto e0, use that its norm is -1, not +1, hence the addition.

    v2 = camera_up_vector + (camera_up_vector'local_metric*e0) .*e0 - (camera_up_vector'local_metric*e1) .*e1

    e2 = v2 ./ sqrt(v2'local_metric*v2)

    e3 = sqrt(-metric_determinant) .* levi

    e3_lower = zeros(Float64,4)

    for l in 1:4
        for u in 1:4
            for v in 1:4
                for p in 1:4
                    e3_lower[p] = e3_lower[p] + levi[l,u,v,p] .* e0[l] .* e1[u] .* e2[v]
                end
            end
        end
    end

    e3 = local_inverse_metric * e3_lower

    e3 = e3 ./ sqrt(e3'local_metric*e3)

    a_array = collect(LinRange(0,1,N_x))
    b_array = collect(LinRange(0,1,N_y))

    alpha_h = angular_pixellation * N_x 
    alpha_v = angular_pixellation * N_y

    println("Your selected vertical, corrected radian range is: "*string(alpha_v))
    println("Your selected horizontal, corrected radian range is: "*string(alpha_h))

    meshgrid_a = ones(N_y) .* a_array'
    meshgrid_b = b_array .* ones(N_x)'

    preliminary_lower_momenta = Vector{MVector{4,Float64}}(undef,N_x*N_y)

    for k in eachindex(meshgrid_a)
        a = meshgrid_a[k]
        b = meshgrid_b[k]
        C = sqrt(1 + (2b-1)^2 * tan(alpha_v/2)^2 + (2a-1)^2 * tan(alpha_h/2)^2 - l_eps)
        
        temp = C .* e0 - e1 - (2b-1) * tan(alpha_v/2) .* e2 - (2a-1) * tan(alpha_h/2) .* e3
        temp .= temp./temp[1]
        lowered_momenta = local_metric * temp
        preliminary_lower_momenta[k] = lowered_momenta
        
        
    end

    outp = Vector{MVector{8,Float64}}(undef,N_x*N_y)

    for k in eachindex(outp)
        outp[k] = MVector{8,Float64}([initial_fourpos; preliminary_lower_momenta[k]])
    end

    return outp
end

function render_image(metric::ADM_metric_container,image_path::String,N_x_cam::Int64,N_y_cam::Int64,
    final_allvector::Vector{MVector{8,Float64}},celestial_sphere_caster::Function,auxillary_colorer::Function)

    quasi_phi, quasi_theta = celestial_sphere_caster(metric,final_allvector)

    #stores two angles (ranging from 0 to 2pi and from 0 to pi)

    celestial_sphere = load(image_path)

    Ny, Nx = size(celestial_sphere)

    output_image = Matrix{eltype(celestial_sphere)}(undef,(N_y_cam,N_x_cam))

    quasi_theta = reshape(quasi_theta, (N_y_cam,N_x_cam))
    quasi_phi = reshape(quasi_phi, (N_y_cam,N_x_cam))

    for j in 1:N_x_cam
        for i in 1:N_y_cam
            y_index = ceil(Int64,quasi_theta[i,j]*Ny/(pi) ) 
            x_index = ceil(Int64,quasi_phi[i,j]*Nx/(2pi) ) 
            
            output_image[i,j] = celestial_sphere[y_index, x_index]
        end
    end

    output_image = auxillary_colorer(metric,final_allvector,output_image)

    return output_image


end

function integrated_redshift_ratio(metric::ADM_metric_container,
    final_allvector::MVector{8,Float64},initial_allvector::MVector{8,Float64})

    #can't take into account the source moving when used in conjungtion with the current ADM integrator

    p_final = metric.numeric_inverse_metric(final_allvector[1:4]) * final_allvector[5:8]
    p_initial = metric.numeric_inverse_metric(initial_allvector[1:4]) * initial_allvector[5:8]

    redshift = p_final[1]/p_initial[1] #or maybe inverse..? not sure

    return redshift 
end

function deterministic_redshift(metric::ADM_metric_container,
    obs_fourposveloc::MVector{8,Float64},source_fourposveloc::MVector{8,Float64})

    #fourposveloc stores variables as [t, x0, x1, x2, 1.0, dx0/dt...]

    dtau_dt2_source = source_fourposveloc[5:8]'metric.numeric_metric[source_fourposveloc[1:4]]*source_fourposveloc[5:8]
    dtau_dt2_obs = obs_fourposveloc[5:8]'metric.numeric_metric[source_fourposveloc[1:4]]*obs_fourposveloc[5:8]

    redshift = sqrt(dtau_dt2_source/dtau_dt2_obs)

    return redshift

end


#module end -.-.-.-.-.-.-
end



using Symbolics, StaticArrays, Plots, Colors, Images

@variables x1::Real, x2::Real, x3::Real, x4::Real

coordinates = @SVector [x1, x2, x3, x4]

c = 1.0
M = 1.0
G = 1.0
r_s = 2.0 * G * M / c^2

sch_g_00 = -(1.0 - r_s / x2) * c^2 
sch_g_11 =  (1.0 - r_s / x2)^(-1.0)
sch_g_22 = x2^2 * sin(x4)^2
sch_g_33 = x2^2 

sch_metric_representation = @SMatrix [
    sch_g_00 0.0 0.0 0.0;
    0.0 sch_g_11 0.0 0.0;
    0.0 0.0 sch_g_22 0.0;
    0.0 0.0 0.0 sch_g_33
]

function SCH_d0_scaler(metric::ADM_raytracing.ADM_metric_container,allvector::Vector{MVector{8,Float64}},def::Float64 = -0.025)
    N_rays = length(allvector)
    outp = zeros(Float64,N_rays)
    for i in 1:N_rays
        outp[i] = def
    end
    return outp
end

function SCH_termination_cause(metric::ADM_raytracing.ADM_metric_container,coord_allvector::Vector{MVector{8, Float64}},
    current_indices::Vector{Int64})
    N_current = length(coord_allvector)
    global_indices_to_del = Vector{Int64}()
    local_indices_to_del = Vector{Int64}()
    
    for i in 1:N_current
        if coord_allvector[i][2] < 2.0 * 1.025 || coord_allvector[i][2] > 30.0
            push!(global_indices_to_del,current_indices[i])
            push!(local_indices_to_del,i)
        end
    end
    return global_indices_to_del, local_indices_to_del
end

function SCH_CS_caster(metric::ADM_raytracing.ADM_metric_container,final_allvector::Vector{MVector{8, Float64}})

    #just cheat using the conservation of angular momentum
    N_rays = length(final_allvector)
    phi = zeros(Float64,N_rays)
    theta = zeros(Float64,N_rays)
    for k in 1:N_rays
        x = cos(final_allvector[k][3]) * sin(final_allvector[k][4])
        y = sin(final_allvector[k][3]) * sin(final_allvector[k][4])
        z = cos(final_allvector[k][4])

        phi[k] = (atan(y,x) + pi) % (2pi)
        theta[k] = acos(z/sqrt(x^2 + y^2 + z^2))
    end

    return phi, theta

end

function SCH_colorer(metric::ADM_raytracing.ADM_metric_container,final_allvectors::Vector{MVector{8, Float64}},image::Matrix{RGBA{N0f8}})
    for i in eachindex(image)
        if final_allvectors[i][2] < 2.0 * 1.025
            
            image[i] = RGBA{N0f8}(0.0,0.0,0.0,1.0)
        end
    end
    return image
end


SCH_ADM = ADM_raytracing.ADM_metric_container(sch_metric_representation,coordinates)

#example of raytracing in SCH spacetime

#=
initial_position = [0.0,10.0,0.0,pi/2]
initial_spatial_veloc = [0.7,0.2,0.0]
init_guess = [@MVector [0.0,10.0,0.0,pi/2,0.0,0.2,0.8,0.0] for k in -10:10]
deviations = [@MVector [0.0,0.0,0.0,0.0,0.0,0.0,k*0.06,0.0] for k in -10:10]

init_conditions = init_guess .+ deviations

SCH_ADM.numeric_u0.(init_conditions,init_conditions)

const timesteps = 2000
raytraced_coords = ADM_raytracing.integrate_ADM_geodesics_RK4_tracked(SCH_ADM,init_conditions,timesteps,SCH_d0_scaler)
n_rays = length(raytraced_coords[1])

final_allvector = ADM_raytracing.integrate_ADM_geodesics_RK4(SCH_ADM,init_conditions,timesteps,SCH_d0_scaler,SCH_termination_cause)

r = zeros(Float64,n_rays,timesteps)
phi = zeros(Float64,n_rays,timesteps)

for n in 1:n_rays
    for t in 1:timesteps
        r[n,t] = raytraced_coords[t][n][2]
        phi[n,t] = raytraced_coords[t][n][3]
    end
end

x = r .* cos.(phi)
y = r .* sin.(phi)
#run the following in the REPL if it doesnt work in VSCODE
plot([x[n,:] for n in 1:n_rays], 
[y[n,:] for n in 1:n_rays])
xlims!(-20, 20)
ylims!(-20, 20)

println("test")
=#

camera_veloc = @MVector [1.0,0.2,0.0,0.0]
camera_pos = @MVector [0.0,15.0,0.0,pi/2]
camera_front = @MVector [0.0, -1.0, 0.0, 0.0]
camera_up = @MVector [0.0,0.0,0.0,1.0]

N_x = 400
N_y = 200
rays_initial_allvector = ADM_raytracing.camera_rays_generator(SCH_ADM,camera_pos,camera_veloc,camera_front,camera_up,0.006,N_x,N_y)
final_allvector = ADM_raytracing.integrate_ADM_geodesics_RK4(SCH_ADM,rays_initial_allvector,6000,SCH_d0_scaler,SCH_termination_cause)
test_image = ADM_raytracing.render_image(SCH_ADM,"raytracing/celestial_spheres/QUASI_CS.png",N_x,N_y,final_allvector,SCH_CS_caster,SCH_colorer)

println("test")