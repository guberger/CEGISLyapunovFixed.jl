function process_PLF_fixed(meth,
                           M, dim, systems, seeds_init,
                           ϵ, tol, δ_min, solvers...; kwargs...)
    output_period = get(kwargs, :output_period, 1)
    level_output = get(kwargs, :level_output, 2)
    output_pad = level_output ≥ 2 ? 4 : -1
    if level_output ≥ 2
        output_period = 1
    end
    depth_max = get(kwargs, :depth_max, -1)
    solverLP = solvers[1]
    solverM = get(solvers, 2, solverLP)

    iter = 0
    depth_rec = 0 # record of max reached depth
    n_aborted = 0
    n_infeasible = 0
    flag = false
    obj_max = Inf
    # nodes_queue = PriorityQueue{Tree,Float64}(Base.Order.Reverse)
    nodes_queue = PriorityQueue{Tree,Float64}()
    nodes = Root()
    for seed_init in seeds_init
        nodes = seed(seed_init)
        enqueue!(nodes_queue, nodes, Inf)
    end

    coeffs_cube = ϵ.*hypercube(dim)
    M0 = length(coeffs_cube)
    M1 = M + M0
    coeffs = [zeros(dim) for i = 1:M1]
    for i = 1:M0
        copyto!(coeffs[i], coeffs_cube[i])
    end
    ζ = 2/ϵ
    meth_cheby = Chebyshev()
    
    while !isempty(nodes_queue)
        iter += 1
        if output_period ≥ 0 && mod(iter, output_period) == 0
            @printf("Iter: %d, front: %d, depth_max: %d, ",
                    iter, length(nodes_queue), depth_rec)
            @printf("infeasible: %d, aborted: %d\n",
                    n_infeasible, n_aborted)
        end
        nodes = dequeue!(nodes_queue)
        depth = length(nodes)

        if level_output ≥ 2
            @printf("|--- depth: %d\n", depth)
        end

        if depth_max ≥ 0 && depth > depth_max
            if level_output ≥ 1
                @printf("|--- Abort branch: max depth (%d) exceeded\n",
                        depth_max)
            end
            n_aborted += 1
            continue
        end
        depth_rec = max(depth_rec, depth)

        δ0, flag = learn_PLF_fixed!(meth_cheby, M0, M, dim, coeffs,
                                    nodes, solverLP, output_pad=output_pad)

        !flag && break
        
        if δ0 < eps(1.0)
            if level_output ≥ 1
                @printf("|--- Infeasible branch: δ0: %f (depth: %d)\n",
                        δ0, depth)
            end
            n_infeasible += 1
            continue
        end

        if meth != meth_cheby
            δ, flag = learn_PLF_fixed!(meth, M0, M, dim, coeffs,
                                       nodes, solverM, output_pad=output_pad)
            flag = flag && δ ≥ 0
            !flag && break
        else
            δ = δ0
        end

        if level_output ≥ 2
            @printf("|--- δ: %f (δ0: %f)\n", δ, δ0)
        end

        if δ < δ_min
            if level_output ≥ 1
                @printf("|--- Abort branch: δ: %f (< %f) (depth: %d)\n",
                    δ, δ_min, depth)
            end
            n_aborted += 1
            continue
        end

        x = _VT_(undef, dim)
        obj_max, flag, i, q, σ = verify_PLF!(M1, dim, x, systems, coeffs,
                                             ζ, solverLP)

        !flag && break

        if level_output ≥ 2
            @printf("|--- deriv_max: %f\n", obj_max)
        end

        obj_max < tol && break

        flow = make_flow(systems, x)
        witness = Witness(flow, i)
        for j = M0+1:M1
            node = Node(witness, j)
            child = grow(nodes, node)
            # enqueue!(nodes_queue, child, δ)
            enqueue!(nodes_queue, child, obj_max) # seems much faster!
        end
    end

    @printf("\nTerminated (flag: %s): max depth: %d, deriv_max: %f\n",
        flag, depth_rec, obj_max)

    return coeffs, collect(Node, nodes), obj_max, flag
end