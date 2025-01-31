"""

Implements the Adaptive Local Knockoff Filter for GWAS data, running on 
window-by-window basis, see Algorithm A3 in Gablenz et al (2024). 

# pseudo-code
# 1. screen SNPs
"""
function gwaw_adaptive(
        data::GWASData; 
        window_width::Int = 1000, 
        m::Int = 1,
        lambdas::Vector{Float64} = estimate_lambdas(data)
    )
    # screen SNPs
    nonzero_idx = screen(data, lambdas, window_width = window_width, m = m)
end

function screen(
        data::GWASData, 
        lambdas::Vector{Float64};
        window_width::Int = 1000, 
    )
    # create windows
    windows = div(data.p, window_width)
    betas = Float64[]
    nonzero_idx = Int[] # length 2p

    @showprogress for window in 1:windows    
        # SNPs in current window
        window_start = (window - 1) * window_width + 1
        window_end = window == windows ? data.p : window * window_width
        idx = window_start:window_end

        # initialize data structure and fit
        data_w = initialize_cloaked_data(data, idx)
        swap!(data_w)
        beta_w = prefit(data_w, lambdas)
        append!(betas, beta_w)

        # screen step (SNP-by-SNP, i.e. ignoring groups)
        p_window = length(idx)
        for j in 1:p_window
            if beta_w[j] != 0 || beta_w[j + p_window] != 0
                push!(nonzero_idx, j + window_start - 1)
                push!(nonzero_idx, j + window_start - 1 + p_window)
            end
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
    Xfull = hcat(data.x, data.xko)

    # lasso
    path = glmnet(Xfull, data.y, lambda = lambdas)
    beta = path.betas[:, end]

    return beta
end
