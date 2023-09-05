from __future__ import annotations

import logging
from typing import Optional, Sequence, Union

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery
from torch.nn import Module
from torch.optim import Optimizer

from flash_attention_softmax_n.surgery.surgery_functions import policy_registry

log = logging.getLogger(__name__)

__all__ = ['AttentionSoftmaxN', 'apply_attention_softmax_n']


def apply_attention_softmax_n(
    model: Module,
    softmax_n_param: float,
    optimizers: Optional[Union[Optimizer, Sequence[Optimizer]]] = None
) -> None:
    """
    Replaces the forward method of SelfAttention with a version that uses softmax_n.

    Example:
    ```python
    import transformers

    from flash_attention_softmax_n.surgery import apply_attention_softmax_n


    model = transformers.AutoModel.from_pretrained('bert-base-uncased')
    apply_attention_softmax_n(model=model, softmax_n_param=1.)

    ```

    :param model: Model to transform.
    :param softmax_n_param: The value of n.
    :param optimizers: Existing optimizers that are bound to `model.parameters()`. Omit this parameters if optimizers will be constructed after calling this function.
    """
    def as_replacement_function(surgery_function):

        def replacement_function(module: Module, module_index: int):
            return surgery_function(module, module_index, softmax_n_param=softmax_n_param)

        return replacement_function

    policies = {
        module_class: as_replacement_function(attention_softmax_n_surgery_function)
        for module_class, attention_softmax_n_surgery_function in policy_registry.items()
    }

    replaced_pairs = module_surgery.replace_module_classes(model, optimizers=optimizers, policies=policies)

    count = len(replaced_pairs)
    if count == 0:
        supported_modules = ''.join(sorted(['\n\t' + c.__module__ + '.' + c.__name__ for c in policy_registry.keys()]))
        log.warning(f'AttentionSoftmaxN had no effect on the model! Support for AttentionSoftmaxN surgery '
                    f'is currently limited to the following classes: {supported_modules}')
    else:
        log.info(f'{count} instances of AttentionSoftmaxN added')


class AttentionSoftmaxN(Algorithm):
    """
    Object that applies attention softmax_n in a Mosaic trainer.

    Example:
    ```python
    import composer
    import transformers

    from flash_attention_softmax_n.surgery import AttentionSoftmaxN


    model = transformers.AutoModel.from_pretrained('bert-base-uncased')
    trainer = composer.trainer.Trainer(
        model=model,
        algorithms=[AttentionSoftmaxN(softmax_n_param=1.)]
    )

    ```
    """
    def __init__(self, softmax_n_param: float) -> None:
        self.softmax_n_param = softmax_n_param
        self._applied = False

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    @staticmethod
    def required_on_load() -> bool:
        return True

    def match(self, event: Event, state: State) -> bool:
        del state  # unused
        return event == Event.INIT and not self._applied

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        del event, logger  # unused
        apply_attention_softmax_n(
            state.model,
            softmax_n_param=self.softmax_n_param,
            optimizers=state.optimizers,
        )
        self._applied = True
