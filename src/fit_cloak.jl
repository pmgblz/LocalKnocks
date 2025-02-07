"""

Implements the Adaptive Local Knockoff Filter for GWAS data, running on 
window-by-window basis, see Algorithm A3 in Gablenz et al (2024). 

# pseudo-code
# for window in windows:
#   1. screen SNPs
#   2. fit lasso with interaction
#   3. 
"""
function gwaw_adaptive(
        data::GWASData,
        interaction_idx::Vector{Int}; # only covariates in here are interacting with SNPs
        window_width::Int = 1000, 
        lambdas::Vector{Float64} = estimate_lambdas(data, kappa_lasso=0.6)
    )
    size(Zint, 1) == data.n || error("Check dimension")

    # create windows
    windows = div(data.p, window_width)
    betas = Float64[]
    nonzero_idx = Int[] # length 2p
    for window in 1:windows
        # screen SNPs in current window
        window_start = (window - 1) * window_width + 1
        window_end = window == windows ? data.p : window * window_width
        nonzero_idx = screen(data, lambdas, window_start, window_end)

        # fit lasso with interaction
        lasso_with_interaction(data, )
    end
end

"""
Fits Lasso with interaction for all SNPs between `window_start` and `window_end`
"""
function lasso_with_interaction(
        data::GWASData, 
        window_start::Int,
        window_end::Int,
        interaction_idx::AbstractVector{Int},
        lambdas::Vector{Float64}
    )
    # create interaction variables
    snp_idx = window_start:window_end
    Xfull = create_interaction(
        @view(data.x[:, snp_idx]), 
        @view(data.z[:, interaction_idx])
    )

    # run Lasso. Note: we are re-using the same lambda path again here. Maybe
    # we should use a different lambda path.
    path = glmnet(Xfull, data.y, lambda = lambdas)
    beta = path.betas[:, end]

    # for each SNP, find which Zs are interacting with it
    # for each Z, find which SNPs are interacting with it
    # interaction = Dict{Int, Vector{Int}}()
end

function screen(
        data::GWASData, 
        lambdas::Vector{Float64},
        window_start::Int,
        window_end::Int
    )
    snp_idx = window_start:window_end

    # initialize data structure and fit
    data_w = initialize_cloaked_data(data, snp_idx)
    swap!(data_w)
    beta_w = prefit(data_w, lambdas)

    # screen step (SNP-by-SNP, i.e. ignoring groups)
    p_window = length(snp_idx)
    nonzero_idx = Int[]
    for j in 1:p_window
        if beta_w[j] != 0 || beta_w[j + p_window] != 0
            push!(nonzero_idx, j + window_start - 1)
            push!(nonzero_idx, j + window_start - 1 + p_window)
        end
    end

    return nonzero_idx
end

"""
    initialize_cloaked_data(data::GWASData, snp_indices::AbstractRange{Int})

Initializes a data structure `CloakedGroupKnockoff` which will contain SNPs specified 
in `snp_indices` and can be fitted by calling `fit()`
"""
function initialize_cloaked_data(data::GWASData, snp_indices::AbstractRange{Int})
    # convert to numeric matrix and scale columns to mean 0 var 1
    xfloat = convert(Matrix{Float64}, @view(data.x.snparray[:, snp_indices]), impute=true)
    zscore!(xfloat, mean(xfloat, dims=1), std(xfloat, dims=1))

    return CloakedGroupKnockoff(xfloat, data.y, data.z)
end

function prefit(data::CloakedGroupKnockoff, lambdas::Vector{T}) where T
    # form design matrix, see TODO
    Xfull = hcat(data.x, data.xko, data.z)

    # lasso
    path = glmnet(Xfull, data.y, lambda = lambdas)
    beta = path.betas[:, end]

    p = data.p
    return beta[1:2p]
end

"""
    create_interaction(X::AbstractMatrix, Zint::AbstractMatrix)

Creates the full matrix of interacting each column of `Zint` to each column of 
`X`. Note we also create interaction between `X` and `1 - Zint` because Matteo 
said to do so when `Zint` is binary, which we assume it is. 
"""
function create_interaction(X::AbstractMatrix, Zint::AbstractMatrix)
    n = size(X, 1)
    length(unique(Zint)) > 2 && error("Zint must be a binary matrix/vector")
    p1, p2 = size(X, 2), size(Zint, 2)
    Xfull = Matrix{Float64}(undef, n, p1 + 2 * p1 * p2)

    # first p1 column is just X
    Xfull[:, 1:p1] .= X

    # interaction of X and Zint
    offset = p1
    for (i, zi) in enumerate(eachcol(Zint))
        window = offset + (i - 1) * p1 + 1:p1 * i + offset
        Xfull[:, window] .= X .* zi
    end

    # interaction of X and 1 - Zint
    offset = p1 + p1 * p2
    for (i, zi) in enumerate(eachcol(Zint))
        window = offset + (i - 1) * p1 + 1:p1 * i + offset
        Xfull[:, window] .= X .* (1 .- zi)
    end

    return Xfull
end
