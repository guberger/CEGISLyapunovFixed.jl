using LinearAlgebra
using JuMP
using HiGHS
using Test
@static if isdefined(Main, :TestLocal)
    include("../src/CEGISLyapunovFixed.jl")
else
    using CEGISLyapunovFixed
end
CPL = CEGISLyapunovFixed

solver = optimizer_with_attributes(HiGHS.Optimizer, "output_flag"=>false)

## Parameters
ϵ = 1e-1
tol = eps(1.0)
D = 2
M = 4
meth = CPL.Chebyshev()

## Tests
domain = zeros(1, D)
fields = [[1.0 0; 0.0 1.0]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

seeds_init = (CPL.Node[],)

δ_min = 0.005
coeffs, nodes, obj_max, flag =
    CPL.process_PLF_fixed(meth, M, D, systems, seeds_init,
                          ϵ, tol, δ_min, solver,
                          depth_max=5,
                          output_period=10,
                          level_output=1)

@testset "process_PLF_fixed Chebyshev: infeasible" begin
    @test flag
    @test obj_max > tol
end

fields = [[-101.1 99; 101 -99.1], [-101.1 -99; -101 -99.1]]
sys = CPL.LinearSystem(domain, fields)
systems = (sys,)

coeffs, nodes, obj_max, flag =
    CPL.process_PLF_fixed(meth, M, D, systems, seeds_init,
                          ϵ, tol, δ_min, solver,
                          depth_max=5,
                          level_output=2)
coeffs_normalized = map(c -> c/norm(c, Inf), coeffs)

@testset "process_PLF_fixed Chebyshev: feasible" begin
    @test flag
    @test obj_max < -eps(1.0)
    for s1 in (-1, +1), s2 in (-1, +1)
        @test [s1, s2] in coeffs_normalized
    end
end

println("\nfinished-----------------------------------------------------------")