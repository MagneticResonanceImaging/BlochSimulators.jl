# Run all tests:
#
#     Pkg.test("BlochSimulators", julia_args=["--check-bounds=auto"])
#
# or only some of the files below, by name:
#
#     Pkg.test("BlochSimulators", julia_args=["--check-bounds=auto"],
#              test_args=["forward_sensitivity", "validation"])
#
# `--check-bounds=auto` runs the tests the way the package is used; see the
# `check_bounds` note in .github/workflows/CI.yml for why that matters here.

using Test
using BlochSimulators

# load general packages
using ComputationalResources
using CUDA
using Distributed
using DistributedArrays
using LinearAlgebra
using StaticArrays
using StructArrays


function make_T₁T₂_structarray(nvoxels)
    T₁ = rand(nvoxels)
    T₂ = 0.1 * T₁
    return @parameters T₁ T₂
end

const TEST_FILES = [
    "operators",
    "precision_and_gpu",
    "magnetization",
    "trajectories",
    "signal",
    "finite_difference",
    "forward_sensitivity",
    "inversion",
]

requested = isempty(ARGS) ? TEST_FILES : ARGS
unknown = setdiff(requested, TEST_FILES)
isempty(unknown) || error("Unknown test file(s) $unknown, choose from $TEST_FILES")

for file in requested
    include("$file.jl")
end
