"""
A `GWASData` is a struct that holds all data

+ `x` is the entire PLINK file, containing all SNPs (.bed) files and their 
    summary statistics. 
+ `y` is the response
+ `z` is the non-genetic covariates.  
"""
struct GWASData{T<:AbstractFloat}
    x::SnpData
    y::Vector{T}
    z::Matrix{T}
    n::Int # sample size
    p::Int # number of SNPs
end

function GWASData(x::SnpData, y::Vector{T}, z::VecOrMat{T}) where T
    n, p = size(x)
    n == length(y) && n == size(z, 1) || error("Dimension mismatch")
    return GWASData(x, y, z, n, p)
end

struct GroupKnockoff{T<:AbstractFloat}
    y::Vector{T} # response
    z::Matrix{T} # non-genetic covariates
    x::Matrix{T} # genetic covariates (within 1 window)
    xko::Matrix{T} # knockoffs of x
    groups::Vector{Int} # group membership
end

"""
    run_group_knockoffs(x::Matrix{T})

Given inidivual level data in `x`, run standard group knockoff algorithm.
"""
function run_group_knockoffs(x::Matrix{T}; seed::Int = 2025) where T
    groups = hc_partition_groups(x, cutoff = 0.5)
    mu = mean(x, dims=1) |> vec
    sigma = estimate_sigma(x)

    # set seed to ensure knockoffs generated are reproducible
    Random.seed!(seed)
    ko = modelX_gaussian_rep_group_knockoffs(x, :maxent, groups, mu, sigma, m=1)
    xko = ko.Xko

    return xko, groups
end

function GroupKnockoff(
        x::Matrix{T}, 
        y::AbstractVector{T}, 
        z::AbstractVecOrMat{T}
    ) where T
    xko, groups = run_group_knockoffs(x)
    return GroupKnockoff(y, z, x, xko, groups)
end

struct CloakedGroupKnockoff{T<:AbstractFloat}
    y::Vector{T} # response
    z::Matrix{T} # non-genetic covariates
    x::Matrix{T}
    xko::Matrix{T}
    groups::Vector{Int} # group membership
    swap_idx::BitMatrix # n by p
end

function CloakedGroupKnockoff(
        x::Matrix{T}, 
        y::AbstractVector{T}, 
        z::AbstractVecOrMat{T};
        seed::Int = 2025
    ) where T
    n, p = size(x)

    # construct group knockoffs
    xko, groups = run_group_knockoffs(x, seed = seed)

    # find indices that should be swapped together
    unique_groups = unique(groups)
    swapped_idx = bitrand(n, p)
    @inbounds for g in unique_groups
        # find indices of all groups that are the same as g
        vec = bitrand(n)
        for j in findall(x -> x == g, groups)
            swapped_idx[:, j] .= vec
        end
    end
    return CloakedGroupKnockoff(y, z, x, xko, groups, swapped_idx)
end

"""
    swap!(data::CloakedGroupKnockoff)

For each sample, randomly swaps the original variable with its knockoffs with 
probability 50%. If a variable within a group is swapped, all variables within
that group must be swapped as well. 
"""
function swap!(data::CloakedGroupKnockoff)
    x, xko, swap_idx = data.x, data.xko, data.swap_idx
    n, p = size(swap_idx)
    @inbounds for j in 1:p
        @simd for i in 1:n
            if swap_idx[i, j]
                x[i, j], xko[i, j] = xko[i, j], x[i, j]
            end
        end
    end
    return nothing
end

# restores real X/Xko identity in 1 local environment blockj
function unswap!(data::CloakedGroupKnockoff, interaction_snps::Vector{Int})
    x, xko, groups, swap_idx = data.x, data.xko, data.groups, data.swap_idx
    n, p = size(swap_idx)
    for snp in interaction_snps
        snps = findall(x -> x == groups[snp], groups)
        for j in snps
            for i in 1:n
                if swap_idx[i, j]
                    x[i, j], xko[i, j] = xko[i, j], x[i, j]
                end
            end
        end
    end
    return nothing
end

# this is ran after we are finished with 1 local environment blockj
function swap!(data::CloakedGroupKnockoff, interaction_snps::Vector{Int})
    unswap!(data, interaction_snps)
end
