using Test
using LocalKnocks

@testset "swap" begin
    # test data
    x = randn(7, 7)
    xko = randn(7, 7)
    x_copy = copy(x)
    xko_copy = copy(xko)
    groups = [1, 2, 3, 1, 1, 2, 3]
    data = LocalKnocks.SwapMatrixPair(x, xko, groups)

    # swap things
    LocalKnocks.swap!(data)

    @test data.is_swapped[1] = true
    for g in unique(groups)
        idx = findall(x -> x == g, groups)
        x1 = @view(data.swap_idx[:, idx[1]]) # col 1 of each unique group
        for i in idx[2:end]
            @test all(x1 .≈ data.swap_idx[:, i]) # test col1 is same as other cols
        end
    end

    @test all(data.x[data.swap_idx] .== xko_copy[data.swap_idx])
    @test all(data.xko[data.swap_idx] .== x_copy[data.swap_idx])

    # swap back
    LocalKnocks.swap!(data)
    @test data.is_swapped[1] == false
    @test all(data.x .== x_copy)
    @test all(data.xko .== xko_copy)
end
