"""
    MXModule

A `MXModule` implement the `BaseModule` API by wrapping a `Symbol` and one or more `Executor` for data parallelization.

MXModule is a basic module that wrap a `Symbol`. It is functionally the same
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
  arch :: SymbolicNode
  base :: BaseModule
  data_provider :: AbstractDataProvider
  params_dirty :: Bool
  context :: Vector{Context}

  arg_params :: Dict{Base.Symbol, NDArray}
  aux_params :: Dict{Base.Symbol, NDArray}

  MXModule(arch :: SymbolicNode) = new(arch)
end
function MXModule(arch :: SymbolicNode; 
                  context::Union{Void, Context, Vector{Context}})
  mod = MXModule(arch)
  if isa(context, Void)
    mod.context = [Context(CPU)]
  elseif isa(context, Context)
    mod.context = [context]
  else
    mod.context = context
  end

end

function bind!(mod :: MXModule, data_provider :: AbstractDataProvider; 
               for_training :: Bool=true,
               inputs_need_grad :: Bool=false, 
               force_rebind :: Bool=false, 
               shared_module :: Union{Void, MXModule}=nothing,
               grad_req :: Symbol=:write)
  if force_rebind
    _reset_bind!(mod)
  end

  if mod.base.binded
    warn("Already binded, ignoring bind!()")
    return mod

  if !for_training
    @assert !inputs_need_grad
  end

  mod.base.for_training = for_training
  mod.base.inputs_need_grad = inputs_need_grad
  mod.base.binded = true

  mod.data_provider = data_provider

  if !isa(shared_module, Void)
    @assert shared_module.base.binded && shared_module.base.params_initialized
    #= shared_group = shared_module.exec_group =#
  else
    shared_group = nothing
  end

  mod.exec_group = DataParallelExecutorGroup(mod.arch, mod.context, mod.data_provider,
                                             for_training=for_training, 
                                             inputs_need_grad=inputs_need_grad,
                                             shared_group=shared_group, 
                                             grad_req=grad_req)

  #= input_types = Dict([x.name => x.dtype for x in mod.data_shapes]) =#

  #= if !isa(mod.label_shapes, Void) =#
  #=   update!(input_types, Dict([x.name => x.dtype for x in mod.label_shapes])) =#
  #= end =#

    #= mod.exec_group = Executor() =#
    #= mod.exec_group = DataParallelExecutorGroup(self._symbol, self._context, =#
    #=                                                  self._work_load_list, self._data_shapes, =#
    #=                                                  self._label_shapes, self._param_names, =#
    #=                                                  for_training, inputs_need_grad, =#
    #=                                                  shared_group, logger=self.logger, =#
    #=                                                  fixed_param_names=self._fixed_param_names, =#
    #=                                                  grad_req=grad_req, input_types=input_types) =#

  if !isa(shared_module, Void)
    mod.base.params_initialized = true
    mod.arg_params = shared_module.arg_params
    mod.aux_params = shared_module.aux_params
  elseif mod.base.params_initialized
    # if the parameters are already initialized, we are re-binding
    # so automatically copy the already initialized params
    #= set_params!(mod, mod.exec_group, mod.arg_params, mod.aux_params) =#
  end

  if !isa(shared_module, Void) && shared_module.optimizer_initialized
    borrow_optimizer!(mod, shared_module)
  end

  return mod
end


function init_params!(mod :: MXModule, initializer :: AbstractInitializer=Uniform(0.01), 
                      arg_params :: Dict{Base.Symbol, NDArray}=Dict{Base.Symbol, NDArray}(),
                      aux_params :: Dict{Base.Symbol, NDArray}=Dict{Base.Symbol, NDArray}(),
                    	allow_missing :: Bool=false, force_init :: Bool=false)
	if mod.base.params_initialized && !force_init
		return mod
  end
  @assert binded(mod), 'call bind! before initializing the parameters'

  if !isdefined(mod, :arg_params) || isempty(mod.arg_params)
    #= param_arrays = [ =#
    #=     nd.zeros(x[0].shape, dtype=x[0].dtype) =#
    #=     for x in self._exec_group.param_arrays =#
    #= ] =#
    #= mod.arg_params = {name:arr for name, arr in zip(self._param_names, param_arrays)} =#
  end

  if !isdefined(mod, :aux_params) || isempty(mod.aux_params)
    #= aux_arrays = [ =#
    #=     nd.zeros(x[0].shape, dtype=x[0].dtype) =#
    #=     for x in self._exec_group.aux_arrays =#
    #= ] =#
    #= self._aux_params = {name:arr for name, arr in zip(self._aux_names, aux_arrays)} =#
  end

  for (k, v) in mod.arg_params
  end

  for (k, v) in mod.aux_params
  end

        def _impl(name, arr, cache):
            """Internal helper for parameter initialization"""
            if cache is not None:
                if name in cache:
                    cache_arr = cache[name]

                    # just in case the cached array is just the target itself
                    if cache_arr is not arr:
                        cache_arr.copyto(arr)
                else:
                    if not allow_missing:
                        raise RuntimeError("%s is not presented" % name)
                    if initializer != None:
                        initializer(name, arr)
            else:
                initializer(name, arr)

        for name, arr in self._arg_params.items():
            _impl(name, arr, arg_params)

        for name, arr in self._aux_params.items():
            _impl(name, arr, aux_params)

  mod.base.params_initialized = true
  mod.params_dirty = false

        # copy the initialized parameters to devices
  #= self._exec_group.set_params(self._arg_params, self._aux_params) =#

  return mod
end

"""
		borrow_optimizer!(module, shared_module)

Borrow optimizer from a shared module. Used in bucketing, where exactly the same 
optimizer (esp. kvstore) is used.

# Arguments
* `module` : MXModule
* `shared_module` : MXModule
"""
function borrow_optimizer!(mod :: MXModule, shared_mod :: MXModule)
  @assert shared_mod.base.optimizer_initialized
  mod.optimizer = shared_mod.optimizer
  mod.kvstore = shared_mod.kvstore
  mod.update_on_kvstore = shared_mod.update_on_kvstore
  mod.updater = shared_mod.updater
  mod.base.optimizer_initialized = true
end

function _reset_bind!(mod :: MXModule)
  mod.base.binded = false
  mod.exec_group = ???
  mod.data_shapes = ???
  mod.label_shapes = ???
end
