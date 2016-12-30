module TestModel
using MXNet
using Base.Test

using ..Main: mlp2

################################################################################
# Test Implementations
################################################################################

function test_basic()
  info("Model::basic")

  mlp = mlp2()
  mlp = mx.LinearRegressionOutput(mlp, name=:lro)
  model = mx.FeedForward(mlp)
  @test isdefined(model, :arch)
  @test model.arch == mlp

  srand(2016)
  mx.srand!(2016)
  x = randn(10, 5)
  data = mx.ArrayDataProvider(:data => x, :lro_label => x, batch_size=5)
  mx._init_model(model, data, mx.UniformInitializer(), false)
  @test isdefined(model, :arg_params)
  @test isdefined(model, :aux_params)
end

function test_saveload()
  info("Model::saveload::mlp2")

  fname_prefix = tempname()

  mlp = mlp2()
  mlp = mx.LinearRegressionOutput(mlp, name=:lro)
  model = mx.FeedForward(mlp)

  srand(2016)
  mx.srand!(2016)
  x = randn(10, 5)
  data = mx.ArrayDataProvider(:data => x, :lro_label => x, batch_size=5)
  mx._init_model(model, data, mx.UniformInitializer(), false)

  mx.save_model(model, fname_prefix)
  @test isfile("$fname_prefix-symbol.json")
  @test isfile("$fname_prefix-data.params")

  params_prefix = "newdata"
  mx.save_model(model, fname_prefix, params_prefix)
  @test isfile("$fname_prefix-symbol.json")
  @test isfile("$fname_prefix-$params_prefix.params")

  model_loaded = mx.load_model(fname_prefix)
  @test mx.to_json(model_loaded.arch) == mx.to_json(model.arch)

  pred = mx.predict(model, data)
  pred_loaded = mx.predict(model_loaded, data)
  @test sum(abs(pred - pred_loaded)) < 1e-6

  rm("$fname_prefix-symbol.json")
  rm("$fname_prefix-data.params")
  rm("$fname_prefix-$params_prefix.params")
end

################################################################################
# Run tests
################################################################################

test_basic()
test_saveload()
end
