"""
    AbstractExecutorGroup

Executor group is a convenient tool for managing a group of executors.
"""
abstract AbstractExecutorGroup

type DataParallelExecutorGroup <: AbstractExecutorGroup
  arch :: SymbolicNode
  context :: Vector{Context}
  for_training :: Bool

  data_provider :: AbstractDataProvider
  input_grad_arrays :: Vector{NDArray}
  shared_data_arrays :: ???
  data_arrays :: Vector{Vector{SlicedNDArray}}
  label_arrays :: Vector{Vector{SlicedNDArray}}
  arg_params
  aux_params

  DataParallelExecutorGroup() = new()
  DataParallelExecutorGroup(arch :: SymbolicNode, context :: Vector{Context}, for_training :: Bool) = new(arch, context, for_training)
end
function DataParallelExecutorGroup(arch :: SymbolicNode, context :: Vector{Context},
                                   data :: AbstractDataProvider;
                                   freeze_param_names :: Union{Void, Vector{Symbol}} = nothing,
                                   inputs_need_grad :: Bool = false,
                                   for_training :: Bool = true,
                                   shared_group :: Union{Void, DataParallelExecutorGroup} = nothing,
                                   grad_req :: GRAD_REQ = GRAD_WRITE)
  exec_group = DataParallelExecutorGroup(arch, context, for_training, freeze_param_names)

  if isa(shared_group, Void)
    exec_group.shared_data_arrays = [Dict{???}() for _ in self.context]
  else
    exec_group.shared_data_arrays = shared_group.shared_data_arrays
  end

  bind_exec!(exec_group, data, shared_group, inputs_need_grad)
end

# TODO use shared_group
# TODO add workload - right now simpliest strategy is used
function bind_exec!(self :: DataParallelExecutorGroup,
                    data_provider :: AbstractDataProvider,
                    shared_group :: Union{Void, DataParallelExecutorGroup},
                    inputs_need_grad)
  self.data_provider = data
  num_dev     = length(self.context)
  batch_size  = get_batch_size(data)
  slices      = _split_inputs(batch_size, num_dev)

  arg_names    = list_arguments(self.arch)
  data_names   = [x[1] for x in provide_data(data)]
  label_names  = [x[1] for x in provide_label(data)]
  input_names  = [data_names; label_names]
  param_names  = setdiff(arg_names, input_names)
  aux_names    = list_auxiliary_states(self.arch)

  grad_req_dict = Dict{Symbol, GRAD_REQ}()
  if isa(freeze_param_names, Void)
    # get grad attribute to allow for freezing
    freeze_param_names = Symbol[]
    for (attr, value) in list_all_attr(self.arch)
      sattr = string(attr)
      if endswith(sattr, "grad") && value == "freeze"
        push!(freeze_param_names, Symbol(sattr[1:end-5]))
      end
    end
  end

  # Needs to correspond to the correct id in the update loop layer idx=1:length(param_names).
  self.freeze_idx = filter(i -> in(param_names[i], freeze_param_names), 1:length(param_names))

  # Setup grad_req as a dictionary
  for param in arg_names
    if param in param_names
      if in(param, freeze_param_names)
        grad_req_dict[param] = GRAD_NOP
      else
        grad_req_dict[param] = grad_req
      end
    elseif param in data_names
      if inputs_need_grad
        grad_req_dict[param] = grad_req
      else
        grad_req_dict[param] = GRAD_NOP
      end
    else
      grad_req_dict[param] = GRAD_NOP
    end
  end
  
  train_execs = Array(Executor, num_dev)
  for i = 1:num_dev
    data_shapes = Dict([k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in provide_data(data)])
    label_shapes = Dict([k => tuple(v[1:end-1]...,length(slices[i])) for (k,v) in provide_label(data)])
    train_execs[i] = simple_bind(self.arch, self.context[i]; grad_req=grad_req_dict, data_shapes..., label_shapes...)
    #= dbg_str = mx.debug_str(train_execs[i]) =#
    #= info(string("TempSpace: ", split(dbg_str, ['\n'])[end-2]..., " on ", self.ctx[i])) =#

    copy_params_from(train_execs[i], self.arg_params, self.aux_params)
  end
  self.execs = train_execs

  # set up input data structures
  self.data_arrays  = [SlicedNDArray[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(train_execs)] for name in data_names]
  self.label_arrays = [SlicedNDArray[(slices[i], exec.arg_dict[name]) for (i,exec) in enumerate(train_execs)] for name in label_names]

  self.param_idx    = filter(i -> in(arg_names[i], param_names), 1:length(arg_names))
  self.name_idx     = filter(i -> in(arg_names[i], data_names), 1:length(arg_names))

  self.param_arrays = [NDArray[exec.arg_arrays[i] for exec in train_execs] for i in param_idx]
  self.grad_arrays  = [NDArray[exec.grad_arrays[i] for exec in train_execs] for i in param_idx]
  self.aux_arrays   = [NDArray[exec.aux_arrays[i] for exec in train_execs] for i = 1:length(aux_names)]

  if inputs_need_grad
    self.input_grad_arrays = [NDArray[exec.grad_arrays[i] for exec in train_execs] for i in name_idx]
  else
    self.input_grad_arrays = []
  end

  return self
end

"""
    forward(exec_group, data_batch, is_train)

Split `data_batch` according to workload and run forward on each devices.

# Arguments
* `data_batch` : DataBatch
  Or could be any object implementing similar interface.
* `is_train` : bool
  The hint for the backend, indicating whether we are during training phase.
  Default is `None`, then the value `self.for_training` will be used.
"""
function forward!(exec_group :: DataParallelExecutorGroup, data_batch, is_train :: Union{Void, Bool} = nothing)

  load_data!(exec_group.data_provider, data_batch, exec_group.data_arrays)
  if isa(is_train, Void)
    is_train = exec_group.for_training
  end
  
  if is_train && !isempty(get_label(data_batch))
    load_label!(exec_group.data, data_batch, label_arrays)
  end

  for exec in exec_group.execs
    forward(exec, is_train=is_train)
  end
   # TODO add callbacks here
end

function output_shapes(exec_group :: DataParallelExecutorGroup)
  outputs = [size(out) for out in exec_group.execs[1].outputs]
  [tuple(key, shape) for key, shape in zip(lis_outputs(exec_group.arch), outputs)]
end
