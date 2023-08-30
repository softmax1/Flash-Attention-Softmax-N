from collections import defaultdict
from functools import partial
from typing import DefaultDict, Tuple, Optional, Set, Dict
from warnings import warn

from torch import Tensor, no_grad, mean
from torch.nn import Module

from flash_attention_softmax_n.analysis.statistics import (
    kurtosis_batch_mean,
    skewness_batch_mean,
    variance_batch_mean,
    kurtosis,
    skewness,
    variance
)


_activation_stat_funcs = {
        'kurtosis': kurtosis_batch_mean,
        'skewness': skewness_batch_mean,
        'variance': variance_batch_mean,
        'mean': lambda x: mean(x).item()
    }


@no_grad()
def save_activations_statistics(
        activations: DefaultDict,
        name: str,
        module: Module,
        inp: Tuple,
        out: Tensor
) -> None:
    """
    PyTorch Forward hook to compute the online average of statistics at each forward pass.
    Mutates specified dict objects with each fwd pass.
    """
    try:
        batch_size = out.shape[0]
        batch_weight = batch_size / (activations[name]['n_samples'] + batch_size)
        activations[name]['n_samples'] += batch_size

        # new_value == (1 - weight) * current_value + weight * update_value
        for stat, func in _activation_stat_funcs.items():
            # Down-weight the current value
            activations[name][stat] *= (1 - batch_weight)
            # Add the weighted new value
            activations[name][stat] += batch_weight * func(out)

    except AttributeError as e:
        warn(f"Unable to compute stats for module {name}. Consider setting or updating `layers_to_save` in `register_activation_hooks`. {e}.", UserWarning)


def _check_name(name: str, layers_to_save: Optional[Set[str]] = None) -> bool:
    return (layers_to_save is None and 'attention.output' in name) or (layers_to_save is not None and name in layers_to_save)


def register_activation_hooks(
        model: Module,
        layers_to_save: Optional[Set[str]] = None
) -> DefaultDict[str, DefaultDict[str, float]]:
    """Registers forward hooks in specified layers.
    Parameters
    ----------
    model:
        PyTorch model
    layers_to_save:
        Module names within ``model`` whose activations we want to save. If None, save all layers

    Returns
    -------
    activations_dict:
        dict of dicts containing activations of specified layers in
        ``layers_to_save``.
    """
    activations_dict = defaultdict(lambda: defaultdict(float))  # Python floats are double-precision, so representing the integer-valued 'n_samples' this way is fine.

    for name, module in model.named_modules():
        if _check_name(name, layers_to_save):
            module.register_forward_hook(
                partial(save_activations_statistics, activations_dict, name)
            )

    return activations_dict


@no_grad()
def compute_weight_statistics(model: Module) -> Dict[str, Dict[str, float]]:
    results = dict()
    for name, param in model.named_parameters():
        results[name] = {
            'n_weights': param.numel(),
            'kurtosis': kurtosis(param).item(),
            'skewness': skewness(param).item(),
            'variance': variance(param).item(),
            'mean': mean(param).item()
        }
    return results
