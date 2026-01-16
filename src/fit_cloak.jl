"""

Implements the Adaptive Local Knockoff Filter for GWAS data, running on 
window-by-window basis, see Algorithm A3 in Gablenz et al (2024). Note the
current implementation only supports interaction on binary covariates. 

# pseudo-code
for window in windows:
  1. cloaking
  2. screen SNPs
  3. fit lasso with interaction
  4. for all covariates, find which SNPs are interacting with it
     4.1: TODO: check the code works if we have >1 interaction variable
  5. run Lasso within each covariate environment
     5.1. reveal identities of interacting SNPs
     5.2. get Ws for each of these
  6. perform knockoff filter

# TODO (lambdas)
I think the lambda vector only needs to be used in step 5, where we use it to get the Ws
for different windows, because only there do we need to make sure the SNPs are penalized
equally across the windows. Lassos in earlier steps are more like screening so they can 
be penalized in other ways (e.g. cross validation)
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
        # create CloakedGroupKnockoff: internally is dense matrices and vectors
        window_start = (window - 1) * window_width + 1
        window_end = window == windows ? data.p : window * window_width
        data_w = initialize_cloaked_data(data, window_start:window_end)
        swap!(data_w)

        # screen SNPs in current window
        screened_snps = screen(data_w, lambdas)

        # fit lasso with interaction. Returns a Dict{Int, Vector{Int}}() where
        # interaction[i] is a Vector{Int} containing variables interacting with zi
        interaction = lasso_with_interaction(data_w, interaction_idx, 
                                             screened_snps, lambdas)

        # run Lasso within each covariate environment
        # See Figure A3 (the 2nd big column) in https://arxiv.org/pdf/2412.02182
        for idx in interaction_idx
            # uncloak all the interct vars
            interacting_snps = interaction[idx]
            unswap!(data_w, interacting_snps)

            # subset to males/females and fit lasso
            zi = data_w.z[:, idx]
            for elem in unique(zi)
                rows = findall(x -> x == elem, zi)
                Ws = local_env_lasso(data_w, lambdas, rows, interacting_snps)
            end

            swap!(...)
        end
    end
end


"""
Fits Lasso with interaction for all SNPs in `data_w`, returns a variable `interaction`
that is a `Dict{Int, Vector{Int}}` where `interaction[i]` contains the snps that 
are interacting with binary indicator `z[:, i]`. 
"""
function lasso_with_interaction(
        data_w::CloakedGroupKnockoff, 
        interaction_idx::AbstractVector{Int},
        screened_snps::AbstractVector{Int},
        lambdas::Vector{Float64}
    )
    # create interaction variables
    x, xko, z = data_w.x, data_w.xko, data_w.z
    non_interacting_idx = setdiff(1:size(z, 2), interaction_idx)
    z_int = z[:, interaction_idx]
    Xfull = create_interaction(hcat(x, xko), z_int)

    # put back z, (1-z), and the non-interacting variables
    Xfull = hcat(Xfull, z[:, interaction_idx], 1 .- z[:, interaction_idx], 
                z[:, non_interacting_idx])

    # run Lasso. Note: we are re-using the same lambda path again here. Maybe
    # we should use a different lambda path. Zihuai we were confused about this, 
    # do you have any suggestion?
    path = glmnet(Xfull, data_w.y, lambda = lambdas)
    beta = path.betas[:, end]

    # for each Z, find which SNPs are interacting with it
    p1, p2 = size(x, 2), size(xko, 2)
    interaction = Dict{Int, Vector{Int}}()
    for i in eachindex(interaction_idx)
        # data looks like: X Xko X*z Xko*z X*(1-z) Xko*(1-z) z (1-z)
        offset1 = p1 + p2                        # X Xko
        offset2 = offset1 + p1 * size(z_int, 2)  # X Xko X*z
        offset3 = offset2 + p2 * size(z_int, 2)  # X Xko X*z Xko*z
        offset4 = offset3 + p1 * size(z_int, 2)  # X Xko X*z Xko*z Xko*(1-z)
        offseti = (i - 1) * p1 # offset for current zi

        non0_idx1 = findall(!iszero, beta[(offseti + offset1 + 1):(offseti + offset1 + p1)])
        non0_idx2 = findall(!iszero, beta[(offseti + offset2 + 1):(offseti + offset2 + p2)])
        non0_idx3 = findall(!iszero, beta[(offseti + offset3 + 1):(offseti + offset3 + p1)])
        non0_idx4 = findall(!iszero, beta[(offseti + offset4 + 1):(offseti + offset4 + p2)])
        
        var = interaction_idx[i]
        interaction[var] = union(non0_idx1, non0_idx2, non0_idx3, non0_idx4)
    end

    return interaction
end

"""
    screen(data_w::CloakedGroupKnockoff, lambdas::Vector{Float64})

Screens SNPs in the current window. If either the original or knockoff SNP is 
non-zero, we keep the index of the original SNP (1 through p).
"""
function screen(
        data_w::CloakedGroupKnockoff, 
        lambdas::Vector{Float64}
    )
    beta_w = prefit(data_w, lambdas)
    p = length(beta_w) >> 1

    # screen step (SNP-by-SNP, i.e. ignoring groups)
    nonzero_idx1 = findall(!iszero, beta_w[1:p])
    nonzero_idx2 = findall(!iszero, beta_w[p+1:2p])
    nonzero_idx = union(nonzero_idx1, nonzero_idx2)
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

"""
    prefit(data::CloakedGroupKnockoff, lambdas::Vector{T}) where T
"""
function prefit(data::CloakedGroupKnockoff, lambdas::Vector{T}) where T
    # form design matrix
    Xfull = hcat(data.x, data.xko, data.z)

    # lasso
    path = glmnet(Xfull, data.y, lambda = lambdas)
    beta = path.betas[:, end]

    p1 = size(data.x, 2)
    p2 = size(data.xko, 2)
    return beta[1:(p1 + p2)] # get beta for SNPs only
end

"""

This is the "local predictive model" (Figure A3 in Gablenz et al 2024) that 
computes Ws across all environments. 
"""
function local_env_lasso(data_w::CloakedGroupKnockoff, lambdas::Vector{T}, 
        rows::Vector{Int}, interacting_snps::Vector{Int}
    ) where T
    p1 = size(data_w.x, 2)
    p2 = size(data_w.xko, 2)
    # these data have been "uncloaked"
    Xfull = @views hcat(data_w.x[rows, :], data_w.xko[rows, :], data_w.z[rows, :])
    # lasso
    path = glmnet(Xfull, data_w.y[rows], lambda = lambdas)
    beta = path.betas[:, end][1:(p1 + p2)] # discard beta for covariates
    # get Ws for groups
    groups = data_w.groups

    # knockoff filter


end

"""
    create_interaction(X::AbstractMatrix, Zint::AbstractMatrix)

Creates the full matrix of interacting each column of `Zint` to each column of 
`X`. Note we also create interaction between `X` and `1 - Zint` because Matteo 
said to do so when `Zint` is binary, which we assume it is. 
"""
function create_interaction(X::AbstractVecOrMat, Zint::AbstractVecOrMat)
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
