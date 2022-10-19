# Chebyshev

function _make_constraints_fixed_chebyshev(model, Deb, coeffs_opt,
                                           coeffs, δ, node)
    x = node.witness.flow.point
    i = node.witness.index
    j = node.index
    c = coeffs[j - Deb]
    if i != j
        if i ≤ Deb
            diff = c - coeffs_opt[i]
            α = 1.0
        else
            diff = c - coeffs[i - Deb]
            α = sqrt(2)
        end
        @constraint(model, dot(x, diff) - α*norm(x)*δ ≥ 0)
    elseif i == j
        for dx in node.witness.flow.grads
            @constraint(model, dot(dx, c) + norm(dx)*δ ≤ 0)
        end
    end
    return nothing
end

function learn_PLF_fixed!(::Chebyshev,
                          Deb, M, dim, coeffs_opt,
                          nodes, solver; output_pad=true)
    padstr = output_pad ≥ 1 ? string("|", repeat(" ", output_pad - 1)) : ""

    if output_pad ≥ 0
        @printf("%sLearning PLF (Chebyshev)\n", padstr)
    end

    if isempty(nodes)
        return Inf, true # δ, flag
    end

    model = Model(solver)
    coeffs = [@variable(model, [1:dim]) for i = 1:M]
    δ = @variable(model)

    for i = 1:M
        @constraint(model, coeffs[i] .≤ +1 - δ)
        @constraint(model, coeffs[i] .≥ -1 + δ)
    end

    for node in nodes
        _make_constraints_fixed_chebyshev(model, Deb, coeffs_opt,
                                          coeffs, δ, node)
    end

    @objective(model, Max, δ)

    optimize!(model)

    δ_opt = value(δ)
    for i = 1:M
        map!(cv -> value(cv), coeffs_opt[Deb + i], coeffs[i])
    end

    if output_pad ≥ 0
        @printf("%s|--- status: %s, %s, %s; δ: %f\n",
                padstr, get_status(model)..., δ_opt)
    end

    flag = primal_status(model) == _RSC_(1)

    return δ_opt, flag
end

# MVE

function _make_constraints_fixed_mve(model, Deb, dim, coeffs_opt,
                                     coeffs, Q, node)
    x = node.witness.flow.point
    i = node.witness.index
    j = node.index
    c = coeffs[j - Deb]
    if i != j
        q = view(Q, :, (j - Deb - 1)*dim+1:(j - Deb)*dim)*x
        if i ≤ Deb
            diff = c - coeffs_opt[i]
        else
            diff = c - coeffs[i - Deb]
            q = q - view(Q, :, (i - Deb - 1)*dim+1:(i - Deb)*dim)*x
        end
        @constraint(model, vcat(dot(x, diff), q) in SecondOrderCone())
    elseif i == j
        Qv = view(Q, :, (j - Deb - 1)*dim+1:(j - Deb)*dim)
        for dx in node.witness.flow.grads
            @constraint(model, vcat(-dot(dx, c), Qv*dx) in SecondOrderCone())
        end
    end
    return nothing
end

function learn_PLF_fixed!(::MVE,
                          Deb, M, dim, coeffs_opt,
                          nodes, solver; output_pad=true)
    padstr = output_pad ≥ 1 ? string("|", repeat(" ", output_pad - 1)) : ""

    if output_pad ≥ 0
        @printf("%sLearning PLF (MVE)\n", padstr)
    end

    if isempty(nodes)
        return Inf, true # δ, flag
    end

    N = M*dim
    model = Model(solver)
    coeffs = [@variable(model, [1:dim]) for i = 1:M]
    δ = @variable(model)
    Q = @variable(model, [1:N,1:N], PSD)
    
    Qup = [Q[i, j] for j = 1:N for i = 1:j]
    @constraint(model, vcat(δ, Qup) in MOI.RootDetConeTriangle(N))

    for i = 1:M
        c = coeffs[i]
        for k = 1:dim
            q = view(Q, :, (i - 1)*dim + k)
            @constraint(model, vcat(1 - c[k], q) in SecondOrderCone())
            @constraint(model, vcat(1 + c[k], q) in SecondOrderCone())
        end
    end

    for node in nodes
        _make_constraints_fixed_mve(model, Deb, dim, coeffs_opt,
                                    coeffs, Q, node)
    end

    @objective(model, Max, δ)

    optimize!(model)

    if has_values(model)
        δ_opt = value(δ)
        for i = 1:M
            map!(cv -> value(cv), coeffs_opt[Deb + i], coeffs[i])
        end
    else
        δ_opt = -1.0
    end

    if output_pad ≥ 0
        @printf("%s|--- status: %s, %s, %s; δ: %f\n",
                padstr, get_status(model)..., δ_opt)
    end

    flag = primal_status(model) == _RSC_(1)

    return δ_opt, flag
end