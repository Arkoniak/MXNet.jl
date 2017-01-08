"""
    MXModule

A `MXModule` implement the `BaseModule` API by wrapping a `Symbol` and one or more `Executor` for data parallelization.

Module is a basic module that wrap a `Symbol`. It is functionally the same
    as the `FeedForward` model, except under the module API.

# Arguments
* `symbol` : Symbol
* `data_names` : list of str
  Default is `('data')` for a typical model used in image classification.
* `label_names` : list of str
  Default is `('softmax_label')` for a typical model used in image
  classification.
* `logger` : Logger
  Default is `logging`.
* `context` : Context or list of Context
  Default is `cpu()`.
* `work_load_list` : list of number
  Default `None`, indicating uniform workload.
* `fixed_param_names`: list of str
  Default `None`, indicating no network parameters are fixed.
"""
type MXModule <: AbstractModule
  
  MXModule(arch :: SymbolicNode) = new(arch)
end

function bind!(mod :: MXModule, data_shapes; 
               label_shapes=nothing, 
               for_training :: Bool=true,
               inputs_need_grad :: Bool=false, 
               force_rebind :: Bool=false, 
               shared_module :: AbstractModule=nothing,
               grad_req :: Symbol=:write)
  if force_rebind
    _reset_bind!(mod)
  end

  if mod.binded
    warn("Already binded, ignoring bind!()")
    return mod

  mod.for_training = for_training
  mod.inputs_need_grad = inputs_need_grad
  mod.binded = true

  if !for_training
    @assert !inputs_need_grad
  end

  # TODO - fix these definitions
  mod.data_shapes = data_shapes
#            [x if isinstance(x, DataDesc) else DataDesc(*x) for x in data_shapes]
  if !isa(label_shapes, Void)
    mod.label_shapes = label_shapes
#                [x if isinstance(x, DataDesc) else DataDesc(*x) for x in label_shapes]
  else
    mod.label_shapes = nothing
  end

  if !isa(shared_module, Void)
    @assert isa(shared_module, AbstractModule) &&
      shared_module.binded && shared_module.params_initialized
    shared_group = shared_module.exec_group
  else
    shared_group = nothing
  end

  input_types = Dict([x.name => x.dtype for x in mod.data_shapes])

  if !isa(mod.label_shapes, Void)
    update!(input_types, Dict([x.name => x.dtype for x in mod.label_shapes]))
  end

    #= mod.exec_group = Executor() =#
    #= mod.exec_group = DataParallelExecutorGroup(self._symbol, self._context, =#
    #=                                                  self._work_load_list, self._data_shapes, =#
    #=                                                  self._label_shapes, self._param_names, =#
    #=                                                  for_training, inputs_need_grad, =#
    #=                                                  shared_group, logger=self.logger, =#
    #=                                                  fixed_param_names=self._fixed_param_names, =#
    #=                                                  grad_req=grad_req, input_types=input_types) =#

  if !isa(shared_module, Void)
    mod.params_initialized = true
    mod.arg_params = shared_module.arg_params
    mod.aux_params = shared_module.aux_params
  elseif mod.params_initialized
    # if the parameters are already initialized, we are re-binding
    # so automatically copy the already initialized params
    set_params!(mod, mod.exec_group, mod.arg_params, mod.aux_params)
  end

  if !isa(shared_module, Void) && shared_module.optimizer_initialized
    borrow_optimizer!(mod, shared_module)
  end

  return mod
end

function _reset_bind!(mod :: MXModule)
  self.binded = false
  self.exec_group = ???
  self.data_shapes = ???
  self.label_shapes = ???
end
