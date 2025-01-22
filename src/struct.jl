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

function GroupKnockoff(
        x::Matrix{T}, 
        y::AbstractVector{T}, 
        z::AbstractVecOrMat{T}
    ) where T
    groups = hc_partition_groups(x, cutoff = 0.5)
    mu = mean(x, dims=1) |> vec
    sigma = estimate_sigma(x)
    ko = modelX_gaussian_rep_group_knockoffs(x, :maxent, groups, mu, sigma, m=1)
    return GroupKnockoff(y, z, x, ko.Xko, groups)
end

struct SwapMatrixPair{T<:AbstractFloat}
    x::Matrix{T}
    xko::Matrix{T}
    swap_idx::BitMatrix
    is_swapped::Vector{Bool}
end

function SwapMatrixPair(x::AbstractMatrix, xko::AbstractMatrix, groups::AbstractVector{Int})
    n, p = size(x)
    n != size(xko, 1) || p != size(xko, 2) && error("Dimension mismatch")
    unique_groups = unique(groups)

    # find indices that should be swapped together
    swapped_idx = bitrand(n, p)
    @inbounds for g in unique_groups
        for j in 1:p
            if j != g && g == groups[j]
                swapped_idx[:, j] .= @view(swapped_idx[:, g])
            end
        end
    end

    is_swapped = [false]

    return SwapMatrixPair(x, xko, swapped_idx, is_swapped)
end

"""
    swap!(data::SwapMatrixPair)

For each sample, randomly swaps the original variable with its knockoffs with 
probability 50%. If a variable within a group is swapped, all variables within
that group must be swapped as well. 
"""
function swap!(data::SwapMatrixPair)
    x, xko, idx = data.x, data.xko, data.swap_idx
    n, p = size(idx)
    @inbounds for j in 1:p
        @simd for i in 1:n
            if idx[i, j]
                x[i, j], xko[i, j] = xko[i, j], x[i, j]
            end
        end
    end
    data.is_swapped[1] = !data.is_swapped[1]
    return nothing
end
