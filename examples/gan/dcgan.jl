using MXNet
import MXNet.mx: provide_data, provide_label, get_batch_size

function get_mnist_data(batch_size=100)
  include(joinpath(dirname(@__FILE__), "..", "common", "mnist-data.jl"))
  return get_mnist_providers(batch_size)
end

function make_dcgan_mnist(ngf=64, ndf=64, nc=3, no_bias=true, fix_gamma=true, eps=1e-5+1e-12)
  gout =  @mx.chain mx.Variable(:rand) =>
          mx.Deconvolution(name=:g1, kernel=(2, 2), num_filter=ngf*4, no_bias=no_bias) =>
          mx.BatchNorm(name=:gbn1, fix_gamma=fix_gamma, eps=eps) =>
          mx.Activation(name=:gact1, act_type=:relu) =>

          mx.Deconvolution(name=:g2, kernel=(4, 4), stride=(2, 2), num_filter=ngf*2, no_bias=no_bias) =>
          mx.BatchNorm(name=:gbn2, fix_gamma=fix_gamma, eps=eps) =>
          mx.Activation(name=:gact2, act_type=:relu) =>

          mx.Deconvolution(name=:g3, kernel=(4, 4), stride=(2, 2), num_filter=ngf, no_bias=no_bias) =>
          mx.BatchNorm(name=:gbn3, fix_gamma=fix_gamma, eps=eps) =>
          mx.Activation(name=:gact3, act_type=:relu) =>

          mx.Deconvolution(name=:g4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=nc, no_bias=no_bias) =>
          mx.Activation(name=:gact4, act_type=:tanh)

  label = mx.Variable(:label)
  dloss = @mx.chain mx.Variable(:data) =>
          mx.Reshape(shape=(28, 28, 1, -1)) =>
          mx.Convolution(name=:d1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf, no_bias=no_bias) =>
          mx.LeakyReLU(name=:dact1, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d2, kernel=(4, 4), stride=(2, 2), num_filter=ndf*2, no_bias=no_bias) =>
          mx.BatchNorm(name=:dbn2, fix_gamma=fix_gamma, eps=eps) =>
          mx.LeakyReLU(name=:dact2, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d3, kernel=(4, 4), stride=(2, 2), num_filter=ndf*4, no_bias=no_bias) =>
          mx.BatchNorm(name=:dbn3, fix_gamma=fix_gamma, eps=eps) =>
          mx.LeakyReLU(name=:dact3, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d4, kernel=(2, 2), num_filter=1, no_bias=no_bias) =>
          mx.Flatten()

  dloss = mx.LogisticRegressionOutput(dloss, name=:dloss, label=label)

  return gout, dloss
end

function make_dcgan_arch(ngf=64, ndf=64, nc=3, no_bias=true, fix_gamma=true, eps=1e-5+1e-12)
  gout =  @mx.chain mx.Variable(:rand) =>
          mx.Deconvolution(name=:g1, kernel=(4, 4), num_filter=ngf*8, no_bias=no_bias) =>
          mx.BatchNorm(name=:gbn1, fix_gamma=fix_gamma, eps=eps) =>
          mx.Activation(name=:gact1, act_type=:relu) =>

          mx.Deconvolution(name=:g2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf*4, no_bias=no_bias) =>
          mx.BatchNorm(name=:gbn2, fix_gamma=fix_gamma, eps=eps) =>
          mx.Activation(name=:gact2, act_type=:relu) =>

          mx.Deconvolution(name=:g3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf*2, no_bias=no_bias) =>
          mx.BatchNorm(name=:gbn3, fix_gamma=fix_gamma, eps=eps) =>
          mx.Activation(name=:gact3, act_type=:relu) =>

          mx.Deconvolution(name=:g4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ngf, no_bias=no_bias) =>
          mx.BatchNorm(name=:gbn4, fix_gamma=fix_gamma, eps=eps) =>
          mx.Activation(name=:gact4, act_type=:relu) =>

          mx.Deconvolution(name=:g5, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=nc, no_bias=no_bias) =>
          mx.Activation(name=:gact5, act_type=:tanh)


  label = mx.Variable(:label)
  dloss = @mx.chain mx.Variable(:data) =>
          mx.Convolution(name=:d1, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf, no_bias=no_bias) =>
          mx.LeakyReLU(name=:dact1, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d2, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf*2, no_bias=no_bias) =>
          mx.BatchNorm(name=:dbn2, fix_gamma=fix_gamma, eps=eps) =>
          mx.LeakyReLU(name=:dact2, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d3, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf*4, no_bias=no_bias) =>
          mx.BatchNorm(name=:dbn3, fix_gamma=fix_gamma, eps=eps) =>
          mx.LeakyReLU(name=:dact3, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d4, kernel=(4, 4), stride=(2, 2), pad=(1, 1), num_filter=ndf*8, no_bias=no_bias) =>
          mx.BatchNorm(name=:dbn4, fix_gamma=fix_gamma, eps=eps) =>
          mx.LeakyReLU(name=:dact4, act_type=:leaky, slope=0.2) =>

          mx.Convolution(name=:d5, kernel=(4, 4), num_filter=1, no_bias=no_bias) =>
          mx.Flatten()

  dloss = mx.LogisticRegressionOutput(dloss, name=:dloss, label=label)

  return gout, dloss
end

symG, symD = make_dcgan_mnist()

####################################################
# RandomDataProvider
####################################################

type RandomDataProvider <: mx.AbstractDataProvider
  batch_size :: Int
  shape :: Tuple{Vararg{Int}}

  data_name :: Symbol
end
RandomDataProvider(batch_size, shape) = RandomDataProvider(batch_size, shape, :rand)

provide_data(provider :: RandomDataProvider) = [(provider.data_name, (provider.shape..., provider.batch_size))]
provide_label(provider :: RandomDataProvider) = []
get_batch_size(provider :: RandomDataProvider) = provider.batch_size

Base.eltype(provider :: RandomDataProvider) = mx.DataBatch
Base.start(provider :: RandomDataProvider) = nothing
Base.done(provider :: RandomDataProvider, state) = false
function Base.next(provider :: RandomDataProvider, state::Void = nothing)
  mx.DataBatch(mx.NDArray[mx.copy(rand(Float32, (provider.shape..., provider.batch_size)), mx.cpu())],
               mx.NDArray[], provider.batch_size), nothing
end

####################################################
# Data
####################################################
ctx = mx.cpu()
batch_size = 100
lr = 0.002
beta1 = 0.5
wd = 0.0
Z = 100

train_data, test_data = get_mnist_data(9)

imshow(x, thr = 0) = join(mapslices(join, (x->x ? 'X': ' ').(x'.> thr), 2), "\n") |> print

#= for batch in mx.eachdatabatch(train_data) =#
#=   imshow(mx.copy(mx.Reshape(batch.data[1], shape=(-1, 1, 28, 28)))[:, :, 5], 0.4) =#
#=   break =#
#= end =#

####################################################
# Module G
####################################################

rnd = RandomDataProvider(batch_size, (Z, 1, 1))
modG = mx.Module.SymbolModule(symG, data_names=[:rand], label_names=Symbol[], context = ctx)
mx.Module.bind(modG, rnd)
mx.Module.init_params(modG, initializer = mx.NormalInitializer(mu = 0.02))
mx.Module.init_optimizer(modG, optimizer=mx.ADAM(lr=lr, beta1=beta1, weight_decay=wd))
mods = [modG]

####################################################
# Module D
####################################################

modD = mx.Module.SymbolModule(symD, data_names=[:data], label_names=[:dloss_label], context = ctx)
mx.Module.bind(modD, train_data, inputs_need_grad=true)
mx.Module.init_params(modD, initializer = mx.NormalInitializer(mu = 0.02))
mx.Module.init_optimizer(modD, optimizer=mx.ADAM(lr=lr, beta1=beta1, weight_decay=wd))
mods = push!(mods, modD)


for (i, batch) in enumerate(mx.eachdatabatch(train_data))
  info(i)
  rbatch = next(rnd)[1]

  mx.Module.forward(modG, rbatch, true)
end













#= rnd = RandomDataProvider(2, (5, 5)) =#
#= i = 0 =#
#= for batch in rnd =#
#=   display(mx.copy(batch.data[1])) =#
#=   i = i + 1 =#
#=   if i > 2 =#
#=     break =#
#=   end =#
#= end =#
