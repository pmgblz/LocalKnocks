module LocalKnocks

using SnpArrays
using Knockoffs
using StatsBase
using Random
using GLMNet
using ProgressMeter
using LinearAlgebra
using DataFrames

export gwas_adaptive

include("struct.jl")
include("fit.jl")
include("fit_cloak.jl")

end
