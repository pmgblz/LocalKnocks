"""
    gwas(data::GWASData)

Does GWAS for all SNPs window-by-window, assuming windows are independent.

# pseudo-code
# create window
# for window in windows:
#     1. initialize struct
#     2. Fit struct
# end
# finally: knockoff filter
"""
function gwas(
        data::GWASData; 
        window_width::Int=1000, 
        m::Int = 1,
        lambdas::Vector{Float64} = estimate_lambdas(data)
    )
    # create windows
    windows = div(data.p, window_width)
    betas = Float64[]
    groups = String[]

    @showprogress for window in 1:windows
        # SNPs in current window
        window_start = (window - 1) * window_width + 1
        window_end = window == windows ? data.p : window * window_width
        idx = window_start:window_end

        # initialize data structure and fit
        data_w = initialize(data, idx)
        beta_w = fit(data_w, lambdas)
        append!(betas, beta_w)

        # keep track of groups
        for g in data_w.groups
            push!(groups, "window$(window)_group$(g)_0") # the last _0 implies this is original variable
        end

        # group membership of knockoff variables
        for k in 1:m, g in data_w.groups
            push!(groups, "window$(window)_group$(g)_$k")
        end
    end

    @assert length(betas) == length(groups) "Beta and group vector length differs!"

    # knockoff filter
    qvalues = filter(betas, groups, m)

    # assemble dataframe to return
    df = assemble_result(data, qvalues, groups)

    return df
end

"""
    fit(data::GroupKnockoff, lambdas::Vector{T})

Fits a Lasso regression on `GroupKnockoff` struct using the lambda sequence 
specified in `lambdas`.

# TODO
Optimize memory usage
"""
function fit(data::GroupKnockoff, lambdas::Vector{T}) where T
    # form design matrix, see TODO
    Xfull = hcat(data.x, data.xko, data.z)

    # lasso
    path = glmnet(Xfull, data.y, lambda = lambdas)
    beta = path.betas[:, end]

    p = data.p
    return beta[1:2p]
end

"""
    initialize(data::GWASData, snp_indices::AbstractRange{Int})

Initializes a data structure `GroupKnockoff` which will contain SNPs specified 
in `snp_indices` and can be fitted by calling `fit()`
"""
function initialize(data::GWASData, snp_indices::AbstractRange{Int})
    # convert to numeric matrix and scale columns to mean 0 var 1
    xfloat = convert(Matrix{Float64}, @view(data.x.snparray[:, snp_indices]), impute=true)
    zscore!(xfloat, mean(xfloat, dims=1), std(xfloat, dims=1))

    return GroupKnockoff(xfloat, data.y, data.z)
end

"""
    estimate_lambdas(data::GWASData)

Estimates Lasso's lambda sequence. This sequence will be used for every window
in `gwas` function. 
"""
function estimate_lambdas(data::GWASData; kappa_lasso = 0.6, m::Int = 1)
    N = data.n
    nsnps = data.p

    # compute Z-scores
    xla = SnpLinAlg{Float64}(data.x.snparray, impute=true, center=true, scale=true)
    yscaled = zscore(data.y)
    z = Transpose(xla) * yscaled ./ sqrt(N)

    # get lambda sequence following Zhaomeng's paper in sec 4.2.2:
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC10925382/ 
    lambdamax = maximum(abs, z) / sqrt(N)
    lambdamin = 0.0001lambdamax
    lambda_path = exp.(range(log(lambdamin), log(lambdamax), length=100)) |> reverse!
    lambda = kappa_lasso * maximum(abs, randn((m+1)*nsnps)) / sqrt(N)
    lambda_path = vcat(lambda_path[findall(x -> x > lambda, lambda_path)], lambda)

    return lambda_path
end

"""
    filter(beta::AbstractVector, groups::AbstractVector, m::Int)

Performs (within-group) knockoff filter. The `beta` vector is assumed to have
estimated effect sizes for both the original variable and the knockoff variables. 
Variables are distinguished based on `groups` vector, where 
"""
function filter(beta::AbstractVector, groups::AbstractVector, m::Int)
    original_idx = findall(x -> endswith(x, "_0"), groups)
    T0 = beta[original_idx]
    Tk = [beta[findall(x -> endswith(x, "_$k"), groups)] for k in 1:m]
    kappa, tau, W = MK_statistics(T0, Tk)
    qvalues = get_knockoff_qvalue(kappa, tau, m, groups=groups)
    return qvalues
end

"""
    assemble_result(data::GWASData, qvalues::AbstractVector, groups::AbstractVector)

Create a dataframe to store the final output of `gwas`.
"""
function assemble_result(
        data::GWASData, 
        qvalues::AbstractVector,
        groups::AbstractVector
    )
    df = copy(data.x.snp_info)
    df[!, "qvalues"] = qvalues

    # also save group membership (remember to filter out knockoff variables)
    df[!, "groups"] = groups[findall(x -> endswith(x, "_0"), groups)]

    return df
end

"""
    get_knockoff_qvalue(κ, τ, m, [groups], [rej_bounds])

Computes the knockoff q-value for each variable. The knockoff q-value is the 
minimum target FDR for a given variable to be selected. For details, see eq 19 
of https://www.nature.com/articles/s41467-022-34932-z and replace the 
knockoff-filter by the within-group knockoff filter proposed in Alg2 of "A
Powerful and Precise Feature-level Filter using Group Knockoffs" by Gu and He (2024).

Note: Code is directly translated from Zihuai's R code here:
https://github.com/biona001/ghostknockoff-gwas-reproducibility/blob/main/he_et_al/GKL_RunAnalysis_All.R#L36
"""
function get_knockoff_qvalue(κ::AbstractVector, τ::AbstractVector, m::Int;
    groups::AbstractVector=collect(1:length(τ)), rej_bounds::Int=10000
    )
    b = sortperm(τ, rev=true)
    c_0 = κ[b] .== 0
    offset = 1 / m
    # calculate ratios for top rej_bounds tau values
    ratio = Float64[]
    temp_0 = 0
    for i in eachindex(b)
        temp_0 = temp_0 + c_0[i]
        temp_1 = i - temp_0
        G_factor = maximum(values(countmap(groups[b][1:i])))
        temp_ratio = (offset*G_factor+offset*temp_1) / max(1, temp_0)
        push!(ratio, temp_ratio)
        i > rej_bounds && break
    end
    # calculate q values for top rej_bounds values
    qvalues = ones(length(τ))
    if any(x -> x > 0, τ)
        index_bound = maximum(findall(τ[b] .> 0))
        for i in eachindex(b)
            temp_index = i:min(length(b), rej_bounds, index_bound)
            length(temp_index) == 0 && continue
            qvalues[b[i]] = minimum(ratio[temp_index])*c_0[i]+1-c_0[i]
            i > rej_bounds && break
        end
        qvalues[qvalues .> 1] .= 1
    end
    return qvalues
end

"""
    estimate_sigma(X::AbstractMatrix; [enforce_psd=true], [min_eigval=1e-5])
    estimate_sigma(X::AbstractMatrix, C::AbstractMatrix; [enforce_psd=true],
        [min_eigval=1e-5])

Estimate LD matrices from data `X`, accounting for covariates `C` if there are any. 
We adopt the method for Pan-UKB described in 
`https://pan-dev.ukbb.broadinstitute.org/docs/ld/index.html#ld-matrices`.
If `enforce_psd=true`, then the correlation matrix will be scaled so that the 
minimum eigenvalue is `min_eigval`.

Code source: 
https://github.com/biona001/GhostKnockoffGWAS/blob/main/src/make_hdf5.jl
"""
function estimate_sigma(X::AbstractMatrix, C::AbstractMatrix;
    enforce_psd::Bool=true, min_eigval::Float64 = 1e-5)
    # check for errors
    n = size(X, 1)
    n == size(C, 1) || error("Sample size in X and C should be the same")

    # pan-ukb routine
    Xc = StatsBase.zscore(X, mean(X, dims=1), std(X, dims=1))
    if size(C, 2) > 1
        Xc .-= C * inv(Symmetric(C' * C)) * (C' * Xc)
    end
    Sigma = Xc' * Xc / n

    # numerical stability
    if enforce_psd
        evals, evecs = eigen(Sigma)
        evals[findall(x -> x < min_eigval, evals)] .= min_eigval
        Sigma = evecs * Diagonal(evals) * evecs'
    end

    # scale to correlation matrix
    StatsBase.cov2cor!(Sigma, sqrt.(diag(Sigma)))

    return Sigma
end
estimate_sigma(X; enforce_psd::Bool=true, min_eigval::Float64 = 1e-5) = 
    estimate_sigma(X, zeros(size(X, 1), 0); enforce_psd=enforce_psd, min_eigval=min_eigval)
