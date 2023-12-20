using Symbolics, StaticArrays

@variables x0, x1, x2, x3

c = 1.0
M = 1
G = 1
r_s = 2 * G * M / c^2

g_00 = -(1.0 - r_s / x1) * c^2 
g_11 =  (1,0 - r_s / x1)^(-1)
g_22 = x1^2 * sin(x3)^2
g_33 = x1^2 

println(g_00)

metric_representation = 