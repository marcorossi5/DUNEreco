import torch
from tqdm.auto import tqdm
from ..abstract_net import AbstractNet
from dunedn.networks.utils import BatchProfiler


def performer_inference_pass(
    test_loader: torch.utils.data.DataLoader,
    network: AbstractNet,
    dev: str,
    verbose: int = 1,
    profiler: BatchProfiler = None,
) -> torch.Tensor:
    """Consumes data through CNN or GCNN network and gives outputs.

    Parameters
    ----------
    test_loader: torch.utils.data.DataLoader
        The inference dataset generator.
    network: AbstractNet
        The denoising network.
    dev: str
        The device hosting the computation.
    verbose: int
        Switch to log information. Defaults to 1. Available options:

        - 0: no logs.
        - 1: display progress bar.

    profiler: BatchProfiler
            The profiler object to record batch inference time.

    Returns
    -------
    output: torch.Tensor
        Denoised data, of shape=(N,1,H,W).
    """
    network.eval()
    network.to(dev)
    outs = []
    wrap = tqdm(test_loader, desc="performer.predict") if verbose else test_loader
    if profiler is not None:
        wrap = profiler.set_iterable(wrap)
    for noisy, _ in wrap:
        out = network(noisy.to(dev)).detach().cpu()
        outs.append(out)
    output = torch.cat(outs)
    network.to("cpu")
    return output
