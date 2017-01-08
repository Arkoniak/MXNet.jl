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
type MXModule <: BaseModule
end

_reset_bind!(self :: MXModule)
  self.binded = false
  self.exec_group = ???
  self._data_shapes = ???
  self._label_shapes = ???
end
