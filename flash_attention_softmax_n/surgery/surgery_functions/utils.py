import inspect
import logging
from typing import Callable, Dict, Optional, Type

import torch

log = logging.getLogger(__name__)

AttentionSoftmaxNReplacementFunction = Callable[[torch.nn.Module, int, float], Optional[torch.nn.Module]]


class PolicyRegistry(Dict[Type[torch.nn.Module], AttentionSoftmaxNReplacementFunction]):
    """
    A registry mapping for AttentionSoftmaxN surgery.
    """

    def register(self, *modules: Type[torch.nn.Module]) -> Callable[[AttentionSoftmaxNReplacementFunction], AttentionSoftmaxNReplacementFunction]:
        """
        This decorator registers mappings from torch modules to their surgery functions.

        Example:
        ```python
        # Replace slow_attention_0 in MyModel with flash_attention_n.
        import torch

        from flash_attention_n import slow_attention_n, flash_attention_n
        from flash_attention_n.surgery.surgery_functions import policy_registry


        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = SlowAttention()

            def forward(self, q, k, v):
                return self.attn(q, k, v, softmax_n_param=0.)


        class SlowAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, q, k, v):
                return slow_attention_n(q, k, v, softmax_n_param=0.)


        @policy_registry.register(SlowAttention)
        def slow_attention_converter(module: torch.nn.Module, module_index: int, softmax_n_param: float) -> torch.nn.Module:
            assert isinstance(module, SlowAttention)
            del module_index  # unused
            module.n = softmax_n_param
            setattr(module, 'forward', MethodType(forward, module))
            return module


        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention_n(q, k, v, softmax_n_param=int(self.n))
        ```
        """
        if len(modules) == 0:
            raise ValueError('Registry decoration without any module class inputs has no effect.')

        def _validate_signature(func: Callable):
            # Necessary to enforce that `func` has a valid signature (i.e. is a AttentionSoftmaxNReplacementFunction)
            signature = inspect.signature(func)
            parameters = signature.parameters
            if len(parameters) != 3:
                raise ValueError(f'Each attention softmax N surgery function must accept 2 arguments, {func} accepts {len(parameters)}')
            ((_, module_param), (_, index_param), (softmax_n_name, softmax_n_param)) = parameters.items()
            if module_param.annotation != torch.nn.Module:
                raise TypeError(f'The first argument of attention softmax N surgery function {func} must be of type "torch.nn.Module"')
            if index_param.annotation != int:
                raise TypeError(f'The second argument of attention softmax N surgery function {func} must be of type "int"')
            if softmax_n_param.annotation != float:
                raise TypeError(f'The third argument of attention softmax N surgery function {func} must be of type "float"')
            if softmax_n_name != 'softmax_n_param':
                raise NameError(f'The third argument of function {func} must be named "softmax_n_param"')

        def _register_module(module: Type[torch.nn.Module], func: Callable) -> None:
            if not issubclass(module, torch.nn.Module):
                raise TypeError(f'Module {module.__name__} is not a subclass of `torch.nn.Module`.')
            if module in self:
                raise ValueError(f'An AttentionSoftmaxNReplacementFunction has already been registered for module {module.__name__}.')
            self[module] = func
            return

        def wrapper(func: AttentionSoftmaxNReplacementFunction) -> AttentionSoftmaxNReplacementFunction:
            _validate_signature(func)
            for module in modules:
                _register_module(module, func)
            return func

        return wrapper


# Initialize the policy registry that AttentionSoftmaxN will reference
policy_registry = PolicyRegistry()
