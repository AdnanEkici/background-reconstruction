from __future__ import annotations

from time import perf_counter

import torch
import yaml


def safe_set_device(device: str) -> str:
    """Safely set the computation device, with fallback to CPU.

    Checks whether CUDA is available. If not, the device is forced to ``"cpu"``.

    Parameters
    ----------
    device : str
        Preferred device string (e.g., ``"cuda:0"``, ``"cpu"``).

    Returns
    -------
    str
        The device string: ``"cuda:X"`` if CUDA is available, otherwise ``"cpu"``.
    """
    if not torch.cuda.is_available():
        return "cpu"
    return device


def read_yaml(yaml_file_path: str) -> dict:
    """Read a YAML file and return its contents.

    Parameters
    ----------
    yaml_file_path : str
        Path to the YAML file to read.

    Returns
    -------
    dict
        Parsed contents of the YAML file.
    """
    with open(yaml_file_path) as file:
        data = yaml.safe_load(file)
        return data


def test_time_benchmark(func):
    """Decorator to benchmark function execution time.

    Wraps a function and records its last execution time in the
    attribute ``last_exec_time`` (in seconds).

    Parameters
    ----------
    func : Callable
        The function to benchmark.

    Returns
    -------
    Callable
        Wrapped function with identical behavior to the original but
        with added execution time tracking.

    Notes
    -----
    - The last measured execution time is stored in
      ``wrapper.last_exec_time`` after each call.
    """

    def wrapper(*args, **kwargs):
        """Execute the wrapped function and record its runtime.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the wrapped function.
        **kwargs : dict
            Keyword arguments passed to the wrapped function.

        Returns
        -------
        Any
            The result of the wrapped function.
        """
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        execution_time = end_time - start_time
        wrapper.last_exec_time = execution_time
        return result

    return wrapper
