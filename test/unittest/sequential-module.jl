module TestSequentialModule
using MXNet
using Base.Test

using ..Main: reldiff

################################################################################
# Utils
################################################################################

function create_seq_modules()
  net1 = @mx.chain mx.Variable(:data) =>
           mx.FullyConnected(name=:fc1, num_hidden=4)
  net2 = @mx.chain mx.Variable(:fc1_output) =>
           mx.FullyConnected(name=:fc2, num_hidden=1) =>
           mx.LinearRegressionOutput(name=:linout)

  m1 = mx.Module.SymbolModule(net1, label_names=Symbol[])
  m2 = mx.Module.SymbolModule(net2, data_names=[:fc1_output], label_names=[:linout_label])

  return m1, m2
end

################################################################################
# Test Implementations
################################################################################

function test_basic()
  info("SequentialModule::basic")

  net1 = @mx.chain mx.Variable(:data) =>
           mx.FullyConnected(name=:fc1, num_hidden=4)
  net2 = @mx.chain mx.Variable(:fc1_output) =>
           mx.FullyConnected(name=:fc2, num_hidden=1) =>
           mx.LinearRegressionOutput(name=:linout)

  m1 = mx.Module.SymbolModule(net1, label_names=Symbol[])
  m2 = mx.Module.SymbolModule(net2, data_names=[:fc1_output], label_names=[:linout_label])
  seq_mod = mx.Module.SequentialModule([:linout_label])
  mx.Module.push!(seq_mod, m1)
  mx.Module.push!(seq_mod, m2, take_labels=true)
  @test !mx.Module.isbinded(seq_mod)
  @test !mx.Module.allows_training(seq_mod)
  @test !mx.Module.isinitialized(seq_mod)
  @test !mx.Module.hasoptimizer(seq_mod)

  @test mx.Module.data_names(seq_mod) == [:data]
  @test mx.Module.output_names(seq_mod) == [:linout_output]

  mx.Module.bind(seq_mod, [(4, 10)], [(1, 10)])
  @test mx.Module.isbinded(seq_mod)
  @test !mx.Module.isinitialized(seq_mod)
  @test !mx.Module.hasoptimizer(seq_mod)

  mx.Module.init_params(seq_mod)
  @test mx.Module.isinitialized(seq_mod)

  mx.Module.init_optimizer(seq_mod)
  @test mx.Module.hasoptimizer(seq_mod)
end

function test_shapes()
  info("SequentialModule::Shapes")

  m1, m2 = create_seq_modules()
  seq_mod = mx.Module.SequentialModule([:linout_label])
  mx.Module.push!(seq_mod, m1)
  mx.Module.push!(seq_mod, m2, take_labels=true)

  mx.Module.bind(seq_mod, [(4, 10)], [(1, 10)])
  @test mx.Module.data_shapes(seq_mod) == Dict(:data => (4, 10))
  @test mx.Module.label_shapes(seq_mod) == Dict(:linout_label => (1, 10))
  @test mx.Module.output_shapes(seq_mod) == Dict(:linout_output => (1, 10))

  m1, m2 = create_seq_modules()
  seq_mod2 = mx.Module.SequentialModule()
  mx.Module.push!(seq_mod2, m1)
  mx.Module.push!(seq_mod2, m2, take_labels = true)

  mx.Module.bind(seq_mod2, [(4, 10)])
  @test isempty(mx.Module.label_shapes(seq_mod2))
end

################################################################################
# Run tests
################################################################################

@testset "  Sequential Module Test" begin
  test_basic()
  test_shapes()
end

end
