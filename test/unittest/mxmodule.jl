module TestMXModule
using MXNet
if VERSION ≥ v"0.5.0-dev+7720"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

using ..Main: mlp2

################################################################################
# Test Implementations
################################################################################

function test_create()
  info("MXModule::create")
  mlp = mlp2()
  mod = mx.MXModule(mlp)

  @test mx.data_names(mod) == [:data]
  @test mx.output_names(mod) == [:fc2_output]
end

################################################################################
# Run tests
################################################################################
@testset "MXModule Test" begin
  test_create()
end
end
