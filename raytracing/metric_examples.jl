include("./ADM_raytracing.jl")
using Symbolics, StaticArrays, Plots, Colors, Images, .ADM_raytracing

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

J = 0.95
a = J/(M*c)
su = x2^2 + a^2 * cos(x4)^2
delta = x2^2 - r_s * x2 + a^2

kerr_g_00 = -(1 - r_s * x2 / su) * c^2
kerr_g_20 = - r_s * x2 * a * sin(x4)^2 * c / su
kerr_g_11 = su/delta 
kerr_g_22 = (x2^2 + a^2 + (r_s * x2 * a^2 * sin(x4)^2)/su ) * sin(x4)^2
kerr_g_33 = su

kerr_metric_representation = @SMatrix [
    kerr_g_00 0.0 kerr_g_20 0.0;
    0.0 kerr_g_11 0.0 0.0;
    kerr_g_20 0.0 kerr_g_22 0.0;
    0.0 0.0 0.0 kerr_g_33
]

v_drive = 0.25
x_s = v_drive * (x1 + 15)
v_s = v_drive

r_alc = sqrt((x2-x_s)^2 + x3^2 + x4^2)
R = 5
o = 3
f_r = (tanh(o*(r_alc + R)) - tanh(o*(r_alc - R)))/(2*tanh(o * R))

alc_00 = (v_s^2 * f_r^2 - 1 )
alc_01 = -v_s * f_r


alc_metric_repr = @SMatrix [
    alc_00 alc_01 0.0 0.0;
    alc_01 1.0 0.0 0.0;
    0.0 0.0 1.0 0.0;
    0.0 0.0 0.0 1.0]

function ALC_d0_scaler(metric::ADM_raytracing.ADM_metric_container,allvector::Vector{MVector{8,Float64}},def::Float64 = -0.025)
    N_rays = length(allvector)
    outp = def * ones(Float64,N_rays)
    return outp

end

function ALC_termination_cause(metric::ADM_raytracing.ADM_metric_container,coord_allvector::Vector{MVector{8, Float64}},
    current_indices::Vector{Int64})
    N_current = length(coord_allvector)
    global_indices_to_del = Vector{Int64}()
    local_indices_to_del = Vector{Int64}()
    
    for i in 1:N_current
        if coord_allvector[i][3]^2 + coord_allvector[i][4]^2 > 40^2 
            push!(global_indices_to_del,current_indices[i])
            push!(local_indices_to_del,i)
        end
    end
    return global_indices_to_del, local_indices_to_del
end

function ALC_colorer(metric::ADM_raytracing.ADM_metric_container,final_allvectors::Vector{MVector{8, Float64}},image::Matrix{RGBA{N0f8}})
    
    return image
end

function ALC_CS_caster(metric::ADM_raytracing.ADM_metric_container,final_allvector::Vector{MVector{8, Float64}})

    #assume that the local metric is sufficently flat
    N_rays = length(final_allvector)
    phi = zeros(Float64,N_rays)
    theta = zeros(Float64,N_rays)
    for k in 1:N_rays
        x = final_allvector[k][6]
        y = final_allvector[k][7]
        z = final_allvector[k][8]

        phi[k] = (atan(y,x) + pi + pi/2) % (2pi)
        theta[k] = acos(z/sqrt(x^2 + y^2 + z^2))
    end

    return phi, theta

end

function SCH_d0_scaler(metric::ADM_raytracing.ADM_metric_container,allvector::Vector{MVector{8,Float64}},def::Float64 = -0.025)
    N_rays = length(allvector)
    outp = def * ones(Float64,N_rays)
    for i in 1:N_rays
        if sin(allvector[i][4])^2 * (allvector[i][2]^2 + 0.95^2) < 0.075^2
            outp[i] = def/160
        end
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

function KERR_termination_cause(metric::ADM_raytracing.ADM_metric_container,coord_allvector::Vector{MVector{8, Float64}},
    current_indices::Vector{Int64})
    N_current = length(coord_allvector)
    global_indices_to_del = Vector{Int64}()
    local_indices_to_del = Vector{Int64}()
    r_event = 1 + sqrt(1-0.95^2)
    for i in 1:N_current
        if coord_allvector[i][2] < r_event * 1.02 || coord_allvector[i][2] > 30.0
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

function KERR_colorer(metric::ADM_raytracing.ADM_metric_container,final_allvectors::Vector{MVector{8, Float64}},image::Matrix{RGBA{N0f8}})
    r_event = 1 + sqrt(1-0.95^2)
    for i in eachindex(image)
        if final_allvectors[i][2] < r_event * 1.02
            
            image[i] = RGBA{N0f8}(0.0,0.0,0.0,1.0)
        end
    end
    return image
end


ALC_ADM = ADM_raytracing.ADM_metric_container(alc_metric_repr,coordinates,true,false)

camera_veloc = @MVector [1.0,0.0,0.0,0.0]
camera_pos = @MVector [0.0,0.0,-15.0,0.0]
camera_front = @MVector [0.0, 0.0, 1.0, 0.0]
camera_up = @MVector [0.0,0.0,0.0,1.0]

N_x = 800
N_y = 400
rays_initial_allvector = ADM_raytracing.camera_rays_generator(ALC_ADM,camera_pos,camera_veloc,camera_front,camera_up,0.006,N_x,N_y)
final_allvector = ADM_raytracing.integrate_ADM_geodesics_RK4(ALC_ADM,rays_initial_allvector,12000,ALC_d0_scaler,ALC_termination_cause)
test_image = ADM_raytracing.render_image(ALC_ADM,"raytracing/celestial_spheres/tracker.png",N_x,N_y,final_allvector,ALC_CS_caster,ALC_colorer)

println("test")