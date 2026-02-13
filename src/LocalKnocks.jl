module LocalKnocks

using SnpArrays
using Knockoffs
using StatsBase
using Random
using GLMNet
using ProgressMeter
using LinearAlgebra
using DataFrames

export gwaw_adaptive

include("struct.jl")
include("fit.jl")
include("fit_cloak.jl")

end
