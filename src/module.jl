"""
    AbstractModule

The abstract super type of all Modules in MXNet.jl

`AbstractModule` defines an API for modules.
A module represents a computation component. The design purpose of a module is that it abstract a computation "machine", that one can run forward, backward, update parameters, etc. We aim to make the APIs easy to use, especially in the case when we need to use imperative API to work with multiple modules (e.g. stochastic depth network).

A module has several states:

* Initial state. Memory is not allocated yet, not ready for computation yet.
* Binded. Shapes for inputs, outputs, and parameters are all known, memory allocated,
  ready for computation.
* Parameter initialized. For modules with parameters, doing computation before initializing
  the parameters might result in undefined outputs.
* Optimizer installed. An optimizer can be installed to a module. After this, the parameters
  of the module can be updated according to the optimizer after gradients are computed
  (forward-backward).

In order for a module to interact with others, a module should be able to report the following information in its raw stage (before binded)

* [`data_names`](@ref): Names of required data.
* [`output_names`](@ref): Names of the defined outputs.

And also the following richer information after binded:

* State information
    * [`isbinded`](@ref): indicating whether the memory buffers needed for computation
    have been allocated.
    * [`allows_training`](@ref): whether the module is binded for training (if binded).
    * [`isinitialized`](@ref): indicating whether the parameters of this module have
      been initialized.
    * [`hasoptimizer`](@ref): indicating whether an optimizers is defined and intialized.
    * [`inputs_need_grad`](@ref): indicating whether gradients with respect to the input data is needed. Might be useful when implementing composition of modules.
* Input/Output information:
  * [`data_shapes`](@ref): a list of `(name, shape)`. In theory, since the memory is allocated, we could directly provide the data arrays. But in the case of data parallelization, the data arrays might not be of the same shape as viewed from the external world.
  * [`label_shapes`](@ref): a list of `(name, shape)`. This might be `[]` if the module does not need labels (e.g. it does not contains a loss function at the top), or a module is not binded for training.
  * [`output_shapes`](@ref): a list of `(name, shape)` for outputs of the module.
* Parameters (for modules with parameters):
  * [`get_params()`](@ref): return a tuple `(arg_params, aux_params)`. Each of those is a dictionary of name to `NDArray` mapping. Those `NDArray` always lives on CPU. The actual parameters used for computing might live on other devices (GPUs), this function will retrieve (a copy of) the latest parameters. Therefore, modifying
  * [`set_params`](@ref): assign parameters to the devices doing the computation.
  * [`init_params`](@ref): a more flexible interface to assign or initialize the parameters.
* Setup:
  * [`bind!`](@ref): prepare environment for computation.
  * [`init_optimizer!`](@ref): install optimizer for parameter updating.
* Computation:
  * [`forward`](@ref): forward operation.
  * [`backward`](@ref): backward operation.
  * [`update`](@ref): update parameters according to installed optimizer.
  * [`get_outputs`](@ref): get outputs of the previous forward operation.
  * [`get_input_grads`](@ref): get the gradients with respect to the inputs computed
  in the previous backward operation.
  * [`update_metric`](@ref): update performance metric for the previous forward
  computed results.
* Optional (mostly for backward compatability):
* [`get_symbol`](@ref): the underlying symbolic graph for this module (if any)
  This property is not necessarily constant. For example, for `BucketingModule`,
  this property is simply the *current* symbol being used. For other modules,
  this value might not be well defined.

When those intermediate-level API are implemented properly, the following
high-level API will be automatically available for a module:
* [`fit`](@ref): train the module parameters on a data set
* [`predict`](@ref): run prediction on a data set and collect outputs
* [`score`](@ref): run prediction on a data set and evaluate performance
* [`forward_backward`](@ref):
"""
abstract AbstractModule

@defstruct ModuleState (
  binded                :: Bool = false,
  for_training          :: Bool = false,
  inputs_need_grad      :: Bool = false,
  params_initialized    :: Bool = false,
  optimizer_initialized :: Bool = false
)

################################################################################
# High Level API
################################################################################

function forward_backward(self :: AbstractModule, data_batch)
  forward(self, data_batch, is_train = true)
  backward(self)
end

# TODO finalize description and types
"""
    score(TODO)

Run prediction on `eval_data` and evaluate the performance according to
        `eval_metric`.

score(self::AbstractModule, eval_data, eval_metric; num_batch, batch_end_callback, reset=true, epoch=0)

# Arguments
* `eval_data`: DataIter
* `eval_metric`: EvalMetric
* `num_batch`: int. Number of batches to run. Default is `None`, indicating run until the `DataIter` finishes.
* `batch_end_callback`: function. Could also be a list of functions.
* `reset`: bool. Default `True`, indicating whether we should reset `eval_data` before starting evaluating.
* `epoch` : int. Default 0. For compatibility, this will be passed to callbacks (if any). During training, this will correspond to the training epoch number.
"""
function score(self :: AbstractModule, eval_data, eval_metric; num_batch=nothing, batch_end_callback=nothing, reset=true, epoch=0)
  @assert isbinded(self) && isinitialized(self)

  reset && reset!(eval_data)
  reset!(eval_metric)

  for (nbatch, eval_batch) in enumerate(eval_data)
    if num_batch !== nothing && nbatch == num_back
      break
    end

    forward(self, eval_batch, is_train=false)
    update_metric(self, eval_metric, label(eval_batch))

    if batch_end_callback !== nothing
      error("Not implemented yet!")
    end

    if score_end_callback !== nothing
      error("Not implemented yet!")
    end

  end
  get(eval_metric)
end

# TODO - convert python code to julian
"""
    iter_predict(TODO...)

Iterate over predictions.
    for pred, i_batch, batch in module.iter_predict(eval_data):
        # pred is a list of outputs from the module
        # i_batch is a integer
        # batch is the data batch from the data iterator

# Arguments
* `eval_data` : DataIter
* `num_batch` : int. Default is `None`, indicating running all the batches in the data iterator.
* `reset` : bool. Default is `True`, indicating whether we should reset the data iter before start doing prediction.
"""
function iter_predict end
#= function iter_predict(self :: AbstractModule, eval_data, num_batch=None, reset=True): =#
#=   @assert self.binded && self.params_initialized =#

#=   if reset =#
#=     reset!(eval_data) =#
#=   end =#

#=   for nbatch, eval_batch in enumerate(eval_data): =#
#=     if num_batch is not None and nbatch == num_batch: =#
#=         break =#
#=     self.forward(eval_batch, is_train=False) =#
#=     pad = eval_batch.pad =#
#=     outputs = [out[0:out.shape[0]-pad] for out in self.get_outputs()] =#

#=   yield (outputs, nbatch, eval_batch) =#
#= end =#

# TODO convert python code to julian
"""
    predict(TODO...)

Run prediction and collect the outputs.

# Arguments
* `eval_data` : DataIter
* `num_batch` : int
  Default is `None`, indicating running all the batches in the data iterator.
* `merge_batches` : bool
  Default is `True`, see the doc for return values.
* `reset` : bool
  Default is `True`, indicating whether we should reset the data iter before start
  doing prediction.
* `always_output_list` : bool
  Default is `False`, see the doc for return values.

# Returns
When `merge_batches` is `True` (by default), the return value will be a list
`[out1, out2, out3]`.  Where each element is concatenation of the outputs for
all the mini-batches. If further that `always_output_list` is `False` (by default),
then in the case of a single output, `out1` is returned instead of `[out1]`.
When `merge_batches` is `False`, the return value will be a nested list like
`[[out1_batch1, out2_batch1], [out1_batch2], ...]`. This mode is useful because
in some cases (e.g. bucketing), the module does not necessarily produce the same
number of outputs.

The objects in the results are `NDArray`s. If you need to work with numpy array,
just call `.asnumpy()` on each of the `NDArray`.
"""
function predict(self::AbstractModule, eval_data;
                 num_batch=nothing, merge_batches=true, reset=true)
  @assert isbinded(self) && isinitialized(self)

  reset && reset!(eval_data)

  for (nbatch, eval_batch) in enumerate(eval_data)
    if num_batch !== nothing && nbatch == num_back
      break
    end
    forward(self, eval_batch, is_train=false)

    outputs = get_outputs(self)

    error("Not yet implemented")
  end
end

# TODO convert python code to julian
"""
    fit!(TODO...)

Train the module parameters.

# Arguments
* train_data : DataIter
* eval_data : DataIter
  If not `None`, will be used as validation set and evaluate the performance
  after each epoch.
* eval_metric : str or EvalMetric
  Default `'acc'`. The performance measure used to display during training.
* epoch_end_callback : function or list of function
  Each callback will be called with the current `epoch`, `symbol`, `arg_params`
  and `aux_params`.
* batch_end_callback : function or list of function
  Each callback will be called with a `BatchEndParam`.
* kvstore : str or KVStore
  Default `'local'`.
* optimizer : str or Optimizer
  Default `'sgd'`
* optimizer_params : dict
  Default `(('learning_rate', 0.01),)`. The parameters for the optimizer constructor.
  The default value is not a `dict`, just to avoid pylint warning on dangerous
  default values.
* eval_end_callback : function or list of function
  These will be called at the end of each full evaluation, with the metrics over
  the entire evaluation set.
* eval_batch_end_callback : function or list of function
  These will be called at the end of each minibatch during evaluation
* initializer : Initializer
  Will be called to initialize the module parameters if not already initialized.
* arg_params : dict
  Default `None`, if not `None`, should be existing parameters from a trained
  model or loaded from a checkpoint (previously saved model). In this case,
  the value here will be used to initialize the module parameters, unless they
  are already initialized by the user via a call to `init_params` or `fit`.
  `arg_params` has higher priority to `initializer`.
* aux_params : dict
  Default `None`. Similar to `arg_params`, except for auxiliary states.
* allow_missing : bool
  Default `False`. Indicate whether we allow missing parameters when `arg_params`
  and `aux_params` are not `None`. If this is `True`, then the missing parameters
  will be initialized via the `initializer`.
* force_rebind : bool
  Default `False`. Whether to force rebinding the executors if already binded.
* force_init : bool
  Default `False`. Indicate whether we should force initialization even if the
  parameters are already initialized.
* begin_epoch : int
  Default `0`. Indicate the starting epoch. Usually, if we are resuming from a
  checkpoint saved at a previous training phase at epoch N, then we should specify
  this value as N+1.
* num_epoch : int
  Number of epochs to run training.
"""
function fit! end
#= function fit!(self :: AbstractModule, train_data, eval_data=None, eval_metric='acc', =#
#=             epoch_end_callback=None, batch_end_callback=None, kvstore='local', =#
#=             optimizer='sgd', optimizer_params=(('learning_rate', 0.01),), =#
#=             eval_end_callback=None, =#
#=             eval_batch_end_callback=None, initializer=Uniform(0.01), =#
#=             arg_params=None, aux_params=None, allow_missing=False, =#
#=             force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None, =#
#=             validation_metric=None, monitor=None) =#

#=   @assert num_epoch is not None, 'please specify number of epochs' =#

#=         self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label, =#
#=                   for_training=True, force_rebind=force_rebind) =#
#=         if monitor is not None: =#
#=             self.install_monitor(monitor) =#
#=         self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params, =#
#=                          allow_missing=allow_missing, force_init=force_init) =#
#=         self.init_optimizer(kvstore=kvstore, optimizer=optimizer, =#
#=                             optimizer_params=optimizer_params) =#

#=         if validation_metric is None: =#
#=             validation_metric = eval_metric =#
#=         if not isinstance(eval_metric, metric.EvalMetric): =#
#=             eval_metric = metric.create(eval_metric) =#

#=         ################################################################################ =#
#=         # training loop =#
#=         ################################################################################ =#
#=         for epoch in range(begin_epoch, num_epoch): =#
#=             tic = time.time() =#
#=             eval_metric.reset() =#
#=             for nbatch, data_batch in enumerate(train_data): =#
#=                 if monitor is not None: =#
#=                     monitor.tic() =#
#=                 self.forward_backward(data_batch) =#
#=                 self.update() =#
#=                 self.update_metric(eval_metric, data_batch.label) =#

#=                 if monitor is not None: =#
#=                     monitor.toc_print() =#

#=                 if batch_end_callback is not None: =#
#=                     batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, =#
#=                                                      eval_metric=eval_metric, =#
#=                                                      locals=locals()) =#
#=                     for callback in _as_list(batch_end_callback): =#
#=                         callback(batch_end_params) =#

#=             # one epoch of training is finished =#
#=             for name, val in eval_metric.get_name_value(): =#
#=                 self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val) =#
#=             toc = time.time() =#
#=             self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic)) =#

#=             # sync aux params across devices =#
#=             arg_params, aux_params = self.get_params() =#
#=             self.set_params(arg_params, aux_params) =#

#=             if epoch_end_callback is not None: =#
#=                 for callback in _as_list(epoch_end_callback): =#
#=                     callback(epoch, self.symbol, arg_params, aux_params) =#

#=             #---------------------------------------- =#
#=             # evaluation on validation set =#
#=             if eval_data: =#
#=                 res = self.score(eval_data, validation_metric, =#
#=                                  score_end_callback=eval_end_callback, =#
#=                                  batch_end_callback=eval_batch_end_callback, epoch=epoch) =#
#=                 #TODO: pull this into default =#
#=                 for name, val in res: =#
#=                     self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val) =#

#=             # end of 1 epoch, reset the data-iter for another epoch =#
#=   train_data.reset() =#
#= end =#

##########################################
##########################################
# Abstract methods
##########################################
##########################################

##########################################
# Symbol information
##########################################
"""
    data_names(module)

A list of names for data required by this module.
"""
function data_names(mod :: AbstractModule)
  throw(MethodError(data_names, (typeof(mod), )))
end

"""
    output_names(module)

A list of names for the outputs of this module.
"""
function output_names(mod :: AbstractModule)
  throw(MethodError(output_names, (typeof(mod), )))
end

################################################################################
# Input/Output information
################################################################################
"""
    data_shapes(module)

A list of (name, shape) pairs specifying the data inputs to this module.  
"""
function data_shapes(mod :: AbstractModule)
  throw(MethodError(data_shapes, (typeof(mod), )))
end

"""
    label_shapes(module)

A list of (name, shape) pairs specifying the label inputs to this module.  If this module does not accept labels -- either it is a module without loss function, or it is not binded for training, then this should return an empty list `[]`.
"""
function label_shapes(mod :: AbstractModule)
  throw(MethodError(label_shapes, (typeof(mod), )))
end

"""
    output_shapes(module)

A list of (name, shape) pairs specifying the outputs of this module.
"""
function output_shapes(mod :: AbstractModule)
  throw(MethodError(output_shapes, (typeof(mod), )))
end

################################################################################
# Parameters of a module
################################################################################
"""
    get_params(module)

Get parameters, those are potentially copies of the the actual parameters used to do computation on the device.

# Returns
`(arg_params, aux_params)`, a pair of dictionary of name to value mapping.
"""
function get_params(mod :: AbstractModule)
  throw(MethodError(get_params, (typeof(mod), )))
end

"""
    init_params!(module; kwargs...)

Initialize the parameters and auxiliary states.

# Arguments
* `module` : `AbstractModule`
* `initializer` : `AbstractInitializer`.  Called to initialize parameters if needed.
* `arg_params` : `Dict{Symbol, NDArray}`.  If not empty, should be a dictionary of existing `arg_params`. Initialization will be copied from that.
* `aux_params` : `Dict{Symbol, NDArray}`.  If not empty, should be a dictionary of existing `aux_params`. Initialization will be copied from that.
* `allow_missing` : `Bool`.  If true, params could contain missing values, and the initializer will be called to fill those missing params.
* `force_init` : `Bool`.  If true, will force re-initialize even if already initialized.
"""
function init_params!(mod; 
    initializer :: AbstractInitializer=UniformInitializer(0.01), 
    arg_params :: Dict{Base.Symbol, NDArray}=Dict{Base.Symbol, NDArray}(),
    aux_params :: Dict{Base.Symbol, NDArray}=Dict{Base.Symbol, NDArray}(),
    allow_missing :: Bool=false, force_init :: Bool=false)
  throw(MethodError(init_params!, (typeof(mod), )))
end

"""
    set_params!(TODO...)

Assign parameter and aux state values.

# Arguments
* arg_params : dict. Dictionary of name to value (`NDArray`) mapping.
* aux_params : dict.  Dictionary of name to value (`NDArray`) mapping.
* allow_missing : bool.  If true, params could contain missing values, and the initializer will be called to fill those missing params.
* force_init : bool.  If true, will force re-initialize even if already initialized.
"""
function set_params!(self :: AbstractModule, arg_params, aux_params; allow_missing=False, force_init=True)
  init_params!(self, initializer=None, arg_params=arg_params, aux_params=aux_params, allow_missing=allow_missing, force_init=force_init)
end

################################################################################
# Computations
################################################################################
"""
    forward(module, data_batch, is_train)

Forward computation.

# Arguments
* data_batch : DataBatch. Could be anything with similar API implemented.
* is_train : bool. Default is `None`, which means `is_train` takes the value of `self.for_training`.
"""
function forward(mod :: AbstractModule, data_batch, is_train=None)
  throw(MethodError(forward, (typeof(mod), )))
end

"""
    backward(module, out_grads)

Backward computation.

# Arguments
* `out_grads` : NDArray or list of NDArray, optional. Gradient on the outputs to be propagated back.  This parameter is only needed when bind is called on outputs that are not a loss function.
"""
function backward(mod :: AbstractModule, out_grads=None)
  throw(MethodError(backward, (typeof(mod), )))
end

"""
    get_outputs(module, merge_mulit_context)

Get outputs of the previous forward computation.

# Arguments
* merge_multi_context : bool
  Default is `True`. In the case when data-parallelism is used, the outputs
  will be collected from multiple devices. A `True` value indicate that we
  should merge the collected results so that they look like from a single
  executor.

# Returns
If `merge_multi_context` is `True`, it is like `[out1, out2]`. Otherwise, it
is like `[[out1_dev1, out1_dev2], [out2_dev1, out2_dev2]]`. All the output
elements are `NDArray`. When `merge_multi_context` is `False`, those `NDArray`
might live on different devices.
"""
function get_outputs(mod :: AbstractModule, merge_multi_context=True)
  throw(MethodError(get_outputs, (typeof(mod), )))
end

"""
    get_input_grad(module, merge_multi_context)

Get the gradients to the inputs, computed in the previous backward computation.

# Arguments
* merge_multi_context : bool
  Default is `True`. In the case when data-parallelism is used, the gradients
  will be collected from multiple devices. A `True` value indicate that we
  should merge the collected results so that they look like from a single
  executor.

# Returns
If `merge_multi_context` is `True`, it is like `[grad1, grad2]`. Otherwise, it
is like `[[grad1_dev1, grad1_dev2], [grad2_dev1, grad2_dev2]]`. All the output
elements are `NDArray`. When `merge_multi_context` is `False`, those `NDArray`
might live on different devices.
"""
function get_input_grads(mod :: AbstractModule, merge_multi_context=True)
  throw(MethodError(get_input_grads, (typeof(mod), )))
end

"""
    update(module)

Update parameters according to the installed optimizer and the gradients computed in the previous forward-backward batch.
"""
function update!(mod :: AbstractModule)
  throw(MethodError(update!, (typeof(mod), )))
end

"""
    update_metric!(module, eval_metric, labels)

Evaluate and accumulate evaluation metric on outputs of the last forward computation.

# Arguments   
* eval_metric : EvalMetric
* labels : list of NDArray
  Typically `data_batch.label`.
"""
function update_metric!(mod :: AbstractModule, eval_metric, labels)
  throw(MethodError(update_metric!, (typeof(mod), )))
end

################################################################################
# module setup
################################################################################
"""
    bind!(TODO...)
Bind the symbols to construct executors. This is necessary before one
can perform computation with the module.

# Arguments
* `data_shapes` : list of (str, tuple)
  Typically is `data_iter.provide_data`.
* `label_shapes` : list of (str, tuple)
  Typically is `data_iter.provide_label`.
* `for_training` : bool
  Default is `True`. Whether the executors should be bind for training.
* `inputs_need_grad` : bool
  Default is `False`. Whether the gradients to the input data need to be computed.
  Typically this is not needed. But this might be needed when implementing composition
  of modules.
* `force_rebind` : bool
  Default is `False`. This function does nothing if the executors are already
  binded. But with this `True`, the executors will be forced to rebind.
* `shared_module` : Module
  Default is `None`. This is used in bucketing. When not `None`, the shared module
  essentially corresponds to a different bucket -- a module with different symbol
  but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
* `grad_req` : str, list of str, dict of str to str
  Requirement for gradient accumulation. Can be 'write', 'add', or 'null'
  (default to 'write').
  Can be specified globally (str) or for each argument (list, dict).
"""
function bind!(mod :: AbstractModule, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None,
             grad_req=:write)
  throw(MethodError(bind!, (typeof(mod), )))
end

"""
    init_optimizer!(TODO...)

Install and initialize optimizers.

# Arguments
* kvstore : str or KVStore
  Default `'local'`.
* optimizer : str or Optimizer
  Default `'sgd'`
* optimizer_params : dict
  Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
  just to avoid pylint warning of dangerous default values.
* force_init : bool
  Default `False`, indicating whether we should force re-initializing the
  optimizer in the case an optimizer is already installed.
"""
function init_optimizer!(mod, kvstore=:local, optimizer=SGD(), force_init=false)
  throw(MethodError(init_optimizer!, (typeof(mod), )))
end

################################################################################
# State information
################################################################################

"""
    isbinded(self::AbstractModule) -> Bool
"""
function isbinded(self::AbstractModule)
  return self.opts.binded
end

"""
    allows_training(self::AbstractModule) -> Bool
"""
function allows_training(self::AbstractModule)
  return self.opts.for_training
end

"""
    isinitialized(self::AbstractModule) -> Bool
"""
function isinitialized(self::AbstractModule)
  return self.opts.params_initialized
end

"""
    hasoptimizer(self::AbstractModule) -> Bool
"""
function hasoptimizer(self::AbstractModule)
  return self.opts.optimizer_initialized
end

"""
    inputs_need_grad(self::AbstractModule) -> Bool
"""
function inputs_need_grad(self :: AbstractModule)
  return self.opts.inputs_need_grad
end

################################################################################
# Optional
################################################################################
"""
    get_symbol(self::AbstractModule) -> Nullable{SymbolicNode}
Returns the associated [`SymbolicNode`](@ref) of the module. It might not be defined or change over time.
"""
function get_symbol(self::AbstractModule)
  return Nullable{SymbolicNode}()
end

include("modules/mxmodule.jl")
include("modules/julia_module.jl")
include("modules/sequential_module.jl")
include("modules/bucketing_module.jl")
