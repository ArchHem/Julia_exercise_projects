
module ADM_raytracing
using Symbolics, StaticArrays, ProgressBars, LinearAlgebra

function levis_civita_generator()
    
end

function numeric_array_generator(matrix,coordinates)
    new_functions = build_function(matrix, coordinates)
    
    allocating_matrix = @inline eval(new_functions[1])
    
    return allocating_matrix
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
        beta_up = gamma_up * beta
        alpha = sqrt(beta'beta_up - metric_representation[1,1])

        # https://iopscience.iop.org/article/10.3847/1538-4365/aac9ca/pdf?fbclid=IwAR0pORzJb6EvCVdTIWo32F6wxhdd3_eQE_-x8afe94Y8dY_2IH_NuNcPiD0

        acceleration_vector =  zeros(Num,8)

        acceleration_vector[1] = 1.0
        acceleration_vector[5] = 0.0
        
        @variables u0::Real, u1::Real, u2::Real, u3::Real

        all_lower_u = [u0, u1, u2, u3]

        lower_spatial_u = [u1, u2, u3]

        implicit_u0 = sqrt(lower_spatial_u'gamma_up*lower_spatial_u + epsilon)/alpha

        u0_generator_output = MVector{8,Num}([([coordinates,[implicit_u0,u1,u2,u3]]...)...])
        

        dx_spatial = (gamma_up * lower_spatial_u) ./ implicit_u0 .- beta_up

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
            
            du_spatial[i] = -alpha * implicit_u0 * alpha_deriv + lower_spatial_u'beta_up_deriv - 1/(2 * implicit_u0) * (lower_spatial_u'gamma_up_deriv*lower_spatial_u)
        end

        du_spatial .= simplify.(du_spatial)

        acceleration_vector[6:8] = du_spatial

        inputs = zeros(Num,8)
        for i in 1:4
            inputs[i] = coordinates[i]
            inputs[4+i] = all_lower_u[i]
        end
        

        numeric_metric = numeric_array_generator(metric_representation,coordinates)

        inverse_numeric_metric = numeric_array_generator(inverse_sym_metric,coordinates)

        acceleration_function = numeric_array_generator(acceleration_vector,inputs)
        
        numeric_u0 = numeric_array_generator(u0_generator_output,inputs)

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

        d0 = dt_scaler(allvector)

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

        d0 = dt_scaler(allvector)

        d1_allvec = multi_acc(allvector)

        d2_allvec = multi_acc(allvector .+ 0.5 .* d0 .* d1_allvec)

        d3_allvec = multi_acc(allvector .+ 0.5 .* d0 .* d2_allvec)

        d4_allvec = multi_acc(allvector .+ d0 .* d3_allvec)

        @. allvector += d0/6 * (d1_allvec + 2 * d2_allvec + 2 * d3_allvec + d4_allvec)

        

        global_del, local_del = terminator_function(allvector, index_tracker)

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
    angular_pixellation::Float64 = 0.1,N_x::Int64 = 200,N_y::Int64 = 200)

    local_metric = metric_binder.numeric_metric(initial_fourpos)
    local_inverse_metric = metric_binder.numeric_inverse_metric(initial_fourpos)

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
    v1 = camera_front_vector .- (camera_front_vector'local_metric*e0) .* e0

    e1 = v1 ./ sqrt(v1'local_metric*v1)

    v2 = camera_up_vector .- (camera_up_vector'local_metric*e0) .- (camera_up_vector'local_metric*e1)

    e2 = v2 ./ sqrt(v2'local_metric*v2)







end



end

#module end -.-.-.-.-.-.-

using Symbolics, StaticArrays, Plots

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

function SCH_d0_scaler(allvector::Vector{MVector{8,Float64}},def::Float64 = -0.025)
    N_rays = length(allvector)
    outp = zeros(Float64,N_rays)
    for i in 1:N_rays
        outp[i] = def
    end
    return outp
end

function SCH_termination_cause(coord_allvector::Vector{MVector{8, Float64}},
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

SCH_ADM = ADM_raytracing.ADM_metric_container(sch_metric_representation,coordinates)

#example of raytracing in SCH spacetime

initial_position = [0.0,10.0,0.0,pi/2]
initial_spatial_veloc = [0.7,0.2,0.0]
init_guess = [@MVector [0.0,10.0,0.0,pi/2,0.0,0.2,0.8,0.0] for k in -10:10]
deviations = [@MVector [0.0,0.0,0.0,0.0,0.0,0.0,k*0.06,0.0] for k in -10:10]

init_conditions = init_guess .+ deviations

initial_allvector = SCH_ADM.numeric_u0.(init_conditions)

const timesteps = 2000
raytraced_coords = ADM_raytracing.integrate_ADM_geodesics_RK4_tracked(SCH_ADM,initial_allvector,timesteps,SCH_d0_scaler)
n_rays = length(raytraced_coords[1])

final_allvector = ADM_raytracing.integrate_ADM_geodesics_RK4(SCH_ADM,initial_allvector,timesteps,SCH_d0_scaler,SCH_termination_cause)

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


