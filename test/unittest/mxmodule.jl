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

  @test !mx.isbinded(mod)
  @test !mx.isinitialized(mod)
  @test !mx.hasoptimizer(mod)
  @test !mx.allows_training(mod)
  @test !mx.inputs_need_grad(mod)
end

function test_bind()
  info("MXModule::bind")

  mlp = mlp2()
  mod = mx.MXModule(mlp)
  
  srand(123456)
  x = rand(mx.DEFAULT_DTYPE, 10, 10)
  data = mx.ArrayDataProvider(:data => x; batch_size = 2)
  
  mx.bind!(mod, data)
  @test mx.isbinded(mod)
  @test mx.allows_training(mod)
  @test !mx.inputs_need_grad(mod)
  @test !mx.isinitialized(mod)
  @test !mx.hasoptimizer(mod)
end

function test_init()
  info("MXModule::init")

  srand(123456)
  x = rand(mx.DEFAULT_DTYPE, 10, 10)
  data = mx.ArrayDataProvider(:data => x; batch_size = 2)
  
  mx.bind!(mod, data)
  mx.init_params!(mod)

  @test mx.isbinded(mod)
  @test mx.allows_training(mod)
  @test !mx.inputs_need_grad(mod)
  @test mx.isinitialized(mod)
  @test !mx.hasoptimizer(mod)
end

function test_optimizer()
  info("MXNet::optimizer")

  srand(123456)
  x = rand(mx.DEFAULT_DTYPE, 10, 10)
  data = mx.ArrayDataProvider(:data => x; batch_size = 2)
  
  mx.bind!(mod, data)
  mx.init_params!(mod)
  mx.init_optimizer!(mod)

  @test mx.isbinded(mod)
  @test mx.allows_training(mod)
  @test !mx.inputs_need_grad(mod)
  @test mx.isinitialized(mod)
  @test mx.hasoptimizer(mod)
end

################################################################################
# Run tests
################################################################################
@testset "MXModule Test" begin
  test_create()
  test_bind()
  test_init()
end
end
