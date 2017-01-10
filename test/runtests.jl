using MXNet
using Base.Test

info("WOWOWOOWOWOWOOWOWOWO")

# run test in the whole directory, latest modified files
# are run first, this makes waiting time shorter when writing
# or modifying unit-tests
function test_dir(dir)
  jl_files = sort(filter(x -> ismatch(r".*\.jl$", x), readdir(dir)), by = fn -> stat(joinpath(dir,fn)).mtime)
  map(reverse(jl_files)) do file
    include("$dir/$file")
  end
end
info("WOWOWOOWOWOWOOWOWOWO")

try
  include(joinpath(dirname(@__FILE__), "common.jl"))
catch y
  info("----------------")
  println(y)
  info(y)
  info("----------------")
end

try
  test_dir(joinpath(dirname(@__FILE__), "unittest"))
catch y
  info("----------------")
  println(y)
  info(y)
  info("----------------")
end

# run the basic MNIST mlp example
#= if haskey(ENV, "CONTINUOUS_INTEGRATION") =#
#=   include(joinpath(Pkg.dir("MXNet"), "examples", "mnist", "mlp-test.jl")) =#
#= end =#
