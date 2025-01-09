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
