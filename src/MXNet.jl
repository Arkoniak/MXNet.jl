__precompile__()

module MXNet

# we put everything in the namespace mx, because there are a lot of
# functions with the same names as built-in utilities like "zeros", etc.
export mx
module mx

using Compat
import Compat.String
import Compat.view

using Formatting

# Functions from base that we can safely extend and that are defined by libmxnet.
import Base: round, ceil, floor, cos, sin, abs, sign, exp, sqrt, exp, log, norm,
             transpose

println("=========================")
info("BASE.jl")
println("BASE.jl")
println("=========================")
include("base.jl")
println("=========================")
info("CONTEXT.jl")
println("CONTEXT.jl")
println("=========================")
include("context.jl")
println("=========================")
info("UTIL.jl")
println("UTIL.jl")
println("=========================")
include("util.jl")

println("=========================")
info("NDARRAY.jl")
println("NDARRAY.jl")
println("=========================")
include("ndarray.jl")
println("=========================")
info("RANDOM.jl")
println("RANDOM.jl")
println("=========================")
include("random.jl")

println("=========================")
info("NAME.jl")
println("NAME.jl")
println("=========================")
include("name.jl")
println("=========================")
info("SYMBOLIC-NODE.jl")
println("SYMBOLIC-NODE.jl")
println("=========================")
include("symbolic-node.jl")
println("=========================")
info("EXECUTOR.jl")
println("EXECUTOR.jl")
println("=========================")
include("executor.jl")

println("=========================")
info("METRIC.jl")
println("METRIC.jl")
println("=========================")
include("metric.jl")
println("=========================")
info("OPTIMIZER.jl")
println("OPTIMIZER.jl")
println("=========================")
include("optimizer.jl")
println("=========================")
info("INITIALIZER.jl")
println("INITIALIZER.jl")
println("=========================")
include("initializer.jl")

println("=========================")
info("IO.jl")
println("IO.jl")
println("=========================")
include("io.jl")
println("=========================")
info("KVSTORE.jl")
println("KVSTORE.jl")
println("=========================")
include("kvstore.jl")

println("=========================")
info("CALLBACK.jl")
println("CALLBACK.jl")
println("=========================")
include("callback.jl")
println("=========================")
info("MODEL.jl")
println("MODEL.jl")
println("=========================")
include("model.jl")

println("=========================")
info("VISUALIZE.jl")
println("VISUALIZE.jl")
println("=========================")
include("visualize.jl")

println("=========================")
info("NN-FACTORY.jl")
println("NN-FACTORY.jl")
println("=========================")
include("nn-factory.jl")
println("=========================")
info("END")
println("END")
println("=========================")


end # mx

end # module MXNet
