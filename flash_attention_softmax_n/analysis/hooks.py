from collections import defaultdict
from functools import partial
from typing import DefaultDict, Tuple, Optional, List, Dict, Union

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
    batch_size = out.shape[0]
    batch_weight = batch_size / (activations[name]['n_samples'] + batch_size)
    activations[name]['n_samples'] += batch_size

    kurtosis_update = kurtosis_batch_mean(out)
    activations[name]['kurtosis'] = (1 - batch_weight) * activations[name]['kurtosis'] + batch_weight * kurtosis_update

    skewness_update = skewness_batch_mean(out)
    activations[name]['skewness'] = (1 - batch_weight) * activations[name]['skewness'] + batch_weight * skewness_update

    variance_update = variance_batch_mean(out)
    activations[name]['variance'] = (1 - batch_weight) * activations[name]['variance'] + batch_weight * variance_update

    activations[name]['mean'] = (1 - batch_weight) * activations[name]['mean'] + batch_weight * mean(out).item()


def register_activation_hooks(
        model: Module,
        layers_to_save: Optional[List[str]] = None
) -> DefaultDict[str, Dict[str, Union[int, float]]]:
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
    activations_dict = defaultdict(lambda: {
        'n_samples': 0,
        'kurtosis': 0.,
        'skewness': 0.,
        'variance': 0.,
        'mean': 0.
    })

    for name, module in model.named_modules():
        if layers_to_save is None or name in layers_to_save:
            module.register_forward_hook(
                partial(save_activations_statistics, activations_dict, name)
            )
    return activations_dict


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
