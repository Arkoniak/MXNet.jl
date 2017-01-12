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
  opts :: BaseModule
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
               freeze_param_names :: Union{Void, Vector{Symbol}} = nothing,
               for_training :: Bool=true,
               inputs_need_grad :: Bool=false, 
               force_rebind :: Bool=false, 
               shared_module :: Union{Void, MXModule}=nothing,
               grad_req :: GRAD_REQ=GRAD_WRITE)
  if force_rebind
    _reset_bind!(mod)
  end

  if mod.opts.binded
    warn("Already binded, ignoring bind!()")
    return mod

  if !for_training
    @assert !inputs_need_grad
  end

  mod.opts.for_training = for_training
  mod.opts.inputs_need_grad = inputs_need_grad
  mod.opts.binded = true

  mod.data_provider = data_provider

  if !isa(shared_module, Void)
    @assert shared_module.opts.binded && shared_module.opts.params_initialized
    shared_group = shared_module.exec_group
  else
    shared_group = nothing
  end

  mod.exec_group = DataParallelExecutorGroup(mod.arch, mod.context, mod.data_provider,
                                             freeze_param_names=freeze_param_names,
                                             for_training=for_training, 
                                             inputs_need_grad=inputs_need_grad,
                                             shared_group=shared_group, 
                                             grad_req=grad_req)

  if !isa(shared_module, Void)
    mod.opts.params_initialized = true
    mod.arg_params = shared_module.arg_params
    mod.aux_params = shared_module.aux_params
  elseif mod.opts.params_initialized
    # if the parameters are already initialized, we are re-binding
    # so automatically copy the already initialized params
    #= set_params!(mod, mod.exec_group, mod.arg_params, mod.aux_params) =#
  end

  if !isa(shared_module, Void) && shared_module.optimizer_initialized
    borrow_optimizer!(mod, shared_module)
  end

  return mod
end


function init_params!(mod :: MXModule, initializer :: AbstractInitializer=UniformInitializer(0.01), 
                      arg_params :: Dict{Base.Symbol, NDArray}=Dict{Base.Symbol, NDArray}(),
                      aux_params :: Dict{Base.Symbol, NDArray}=Dict{Base.Symbol, NDArray}(),
                    	allow_missing :: Bool=false, force_init :: Bool=false)
	if mod.opts.params_initialized && !force_init
		return mod
  end
  @assert mod.opts.binded, 'call bind! before initializing the parameters'

  if !isdefined(mod, :arg_params) || isempty(mod.arg_params)
    mod.arg_params = Dict(map((x) -> x[1] => zeros(size(x[2])), mod.exec_group.arg_params))
  end

  if !isdefined(mod, :aux_params) || isempty(mod.aux_params)
    mod.aux_params = Dict(map((x) -> x[1] => zeros(size(x[2])), mod.exec_group.aux_params))
  end

  for (name, arr) in mod.arg_params
    cache = arg_params
    if !isempty(cache)
      if haskey(cache, name)
        # TODO check, may be it should be copy here, not equality
        mod.arg_params[name] = cache[name]
      else
        if !allow_missing
          ???
          throw error "$name is not presented"
        end

        init(initializer, name, arr)
      end
    else
      init(initializer, name, arr)
    end
  end

  for (k, v) in mod.aux_params
    cache = aux_params
    if !isempty(cache)
      if haskey(cache, name)
        # TODO check, may be it should be copy here, not equality
        mod.aux_params[name] = cache[name]
      else
        if !allow_missing
          ???
          throw error "$name is not presented"
        end

        init(initializer, name, arr)
      end
    else
      init(initializer, name, arr)
    end
  end

  mod.opts.params_initialized = true
  mod.params_dirty = false

  # copy the initialized parameters to devices
  set_params!(mod.exec_group, mod.arg_params, mod.aux_params)

  return mod
end

function _create_kvstore(kvstore :: KVStore, num :: Int, arg_params)
  kvstore, true
end
# TODO add description
function init_optimizer!(mod :: MXModule, optimizer, force_init :: Bool = false, kvstore)
  @assert mod.opts.binded && mod.opts.params_initialized

  if mod.opts.optimizer_initialized && !force_init
    warn("Optimizer already initialized, ignoring...")
    return mod
  end

  # TODO initialize KV store
  # setup kvstore
  kvstore, update_on_kvstore = _create_kvstore(kvstore, length(mod.ctx), mod.arg_params)

  mod.optimizer = optimizer
  mod.kvstore = kvstore
  mod.update_on_kvstore = update_on_kvstore
  mod.opts.optimizer_initialized = true

  op_state = OptimizationState(batch_size)
  optimizer.state = op_state

  if !isa(kvstore, Void)
    if update_on_kvstore
      set_optimizer(kvstore, optimizer)
    end

    info("Initializing KVStore...")
    # init kv with gradients
    for idx = 1:length(param_arrays)
      param_on_devs = param_arrays[idx]

      init!(kvstore, idx, self.arg_params[param_names[idx]])

      if update_on_kvstore
        # pull weights back
        pull!(kvstore, idx, param_on_devs, priority=-idx)
      end
    end
  end
  
  if !isa(mod.preload_opt_states, Void)
    load_optimizer_states!(mod, mod.preload_opt_states)
    mod.preload_opt_states = nothing
  end

  return mod
end

# TODO add description
"""
    forward(module, data_batch, is_train)

Forward computation.
# Arguments

* `data_batch` : DataBatch
Could be anything with similar API implemented.
* `is_train` : bool
  Default is `None`, which means `is_train` takes the value of `self.for_training`.
"""

function forward(mod :: MXModule, data_batch, is_train :: Bool = false)
  @assert mod.opts.binded && mod.opts.params_initialized

  forward(mod.exec_group, data_batch, is_train)
end

"""
    backward(module, out_grads)

Backward computation.
# Arguments
* out_grads : NDArray or list of NDArray, optional
  Gradient on the outputs to be propagated back.
  This parameter is only needed when bind is called
  on outputs that are not a loss function.
"""
function backward(mod :: MXModule, out_grads=nothing)
  @assert mod.opts.binded && mod.opts.params_initialized

  backward(mod.exec_group, out_grads=out_grads)
end


"""
    update!(mod)

Update parameters according to the installed optimizer and the gradients computed
in the previous forward-backward batch.
"""
function update!(mod :: MXModule)
  @assert mod.opts.binded && mod.opts.params_initialized && mod.opts.optimizer_initialized

  mod.params_dirty = true
  if mod.update_on_kvstore
    _update_params_on_kvstore(mod.kvstore,
                              self.exec_group.param_arrays,
                              self.exec_group.grad_arrays)
  else
    _update_params(mod.kvstore,
                   mod.exec_group.param_arrays,
                   mod.exec_group.grad_arrays,
                   updater=mod.updater,
                   num_device=length(mod.context))
  end
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
  @assert shared_mod.opts.optimizer_initialized
  mod.optimizer = shared_mod.optimizer
  mod.kvstore = shared_mod.kvstore
  mod.update_on_kvstore = shared_mod.update_on_kvstore
  mod.updater = shared_mod.updater
  mod.opts.optimizer_initialized = true
end

function _reset_bind!(mod :: MXModule)
  mod.opts.binded = false
  mod.exec_group = DataParallelExecutorGroup()
end

"""
    load_optimizer_states(module, filename)

Load optimizer (updater) state from file

# Arguments
* `fname` : str
  Path to input states file.
"""
function load_optimizer_states(mod :: MXModule, fname :: AbstractString)
  @assert mod.opts.optimizer_initialized

  if mod.update_on_kvstore
    load_optimizer_states(mod.kvstore, fname)
  else
    set_states(mod.updater, read(fname))
  end
end
