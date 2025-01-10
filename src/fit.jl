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
function gwas(data::GWASData; window_width::Int=1000)
    # create windows
    windows = div(data.p, window_width)
    lambdas = estimate_lambdas(data)
    beta = Float64[]

    @showprogress for window in 1:windows
        # SNPs in current window
        window_start = (window - 1) * window_width + 1
        window_end = window == windows ? data.p : window * window_width
        idx = window_start:window_end

        # initialize data structure and fit
        data_w = initialize(data, idx)
        beta_w = fit(data_w, lambdas = lambdas)
        # append!(beta, beta_w)
    end
end

"""
    fit(data::GroupKnockoff, lambdas::Vector{T})

Fits a Lasso regression on `GroupKnockoff` struct using the lambda sequence 
specified in `lambdas`.
"""
function fit(data::GroupKnockoff, lambdas::Vector{T}) where T
    # todo
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
    yscaled = zscore(y)
    z = Transpose(xla) * yscaled ./ sqrt(N)

    # get lambda sequence following 
    lambdamax = maximum(abs, z) / sqrt(N)
    lambdamin = 0.0001lambdamax
    lambda_path = exp.(range(log(lambdamin), log(lambdamax), length=100)) |> reverse!
    lambda = kappa_lasso * maximum(abs, randn((m+1)*nsnps)) / sqrt(N)
    lambda_path = vcat(lambda_path[findall(x -> x > lambda, lambda_path)], lambda)

    return lambda_path
end