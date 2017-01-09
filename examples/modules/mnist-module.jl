using MXNet

# define MLP
mlp = @mx.chain mx.Variable(:data)             =>
  mx.FullyConnected(name=:fc1, num_hidden=128) =>
  mx.Activation(name=:relu1, act_type=:relu)   =>
  mx.FullyConnected(name=:fc2, num_hidden=64)  =>
  mx.Activation(name=:relu2, act_type=:relu)   =>
  mx.FullyConnected(name=:fc3, num_hidden=10)  =>
  mx.SoftmaxOutput(name=:softmax)

# data provider
batch_size = 100
include("mnist-data.jl")
train_provider, eval_provider = get_mnist_providers(batch_size)

################################################################################
# Intermediate-level API
################################################################################
mod = mx.MXModule(softmax)
#= bind!(mod, data_shapes=train_dataiter.provide_data, label_shapes=train_dataiter.provide_label) =#
bind!(mod, data_provider=train_provider)
init_params!(mod)

init_optimizer!(mod, optimizer_params={'learning_rate':0.01, 'momentum': 0.9})
metric = mx.metric.create('acc')

for i_epoch in range(n_epoch):
  for i_iter, batch in enumerate(each_batch(train_provider))
    mx.forward(mod, batch)
    mx.update_metric(mod, metric, batch.label)

    mx.backward(mod)
    mx.update(mod)
  end

  for name, val in metric.get_name_value():
    print('epoch %03d: %s=%f' % (i_epoch, name, val))
  end
  reset!(metric)
  reset!(train_dataiter)
end


#= ################################################################################ =#
#= # High-level API =#
#= ################################################################################ =#
#= logging.basicConfig(level=logging.DEBUG) =#
#= train_dataiter.reset() =#
#= mod = mx.mod.Module(softmax) =#
#= mod.fit(train_dataiter, eval_data=val_dataiter, =#
#=         optimizer_params={'learning_rate':0.01, 'momentum': 0.9}, num_epoch=n_epoch) =#

#= # prediction iterator API =#
#= for preds, i_batch, batch in mod.iter_predict(val_dataiter): =#
#=     pred_label = preds[0].asnumpy().argmax(axis=1) =#
#=     label = batch.label[0].asnumpy().astype('int32') =#
#=     if i_batch % 20 == 0: =#
#=         print('batch %03d acc: %.3f' % (i_batch, (label == pred_label).sum() / float(len(pred_label)))) =#

#= # a dummy call just to test if the API works for merge_batches=True =#
#= preds = mod.predict(val_dataiter) =#

#= # perform prediction and calculate accuracy manually =#
#= preds = mod.predict(val_dataiter, merge_batches=False) =#
#= val_dataiter.reset() =#
#= acc_sum = 0.0; acc_cnt = 0 =#
#= for i, batch in enumerate(val_dataiter): =#
#=     pred_label = preds[i][0].asnumpy().argmax(axis=1) =#
#=     label = batch.label[0].asnumpy().astype('int32') =#
#=     acc_sum += (label == pred_label).sum() =#
#=     acc_cnt += len(pred_label) =#
#= print('validation Accuracy: %.3f' % (acc_sum / acc_cnt)) =#

#= # evaluate on validation set with a evaluation metric =#
#= mod.score(val_dataiter, metric) =#
#= for name, val in metric.get_name_value(): =#
#= print('%s=%f' % (name, val)) =#
