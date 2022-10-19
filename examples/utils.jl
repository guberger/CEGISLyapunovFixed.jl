using LinearAlgebra
using PyCall
const spatial = pyimport_conda("scipy.spatial", "scipy")

function compute_vertices_2d(coeffs)
    A = zeros(length(coeffs), 3)
    for (i, c) in enumerate(coeffs)
        for k = 1:2
            A[i, k] = c[k]
        end
        A[i, 3] = -1
    end
    x = zeros(2)
    hs = spatial.HalfspaceIntersection(A, x)
    points = [collect(point_) for point_ in eachrow(hs.intersections)]
    ch = spatial.ConvexHull(points)
    return [ch.points[i + 1, :] for i in ch.vertices]
end