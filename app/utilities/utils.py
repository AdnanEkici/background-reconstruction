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


def iou(boxA, boxB):
    """Compute the Intersection-over-Union (IoU) between two bounding boxes.

    IoU is defined as the area of intersection divided by the area
    of the union of the two bounding boxes.

    Parameters
    ----------
    boxA : tuple[int, int, int, int]
        Bounding box A in format ``(x, y, width, height)``.
    boxB : tuple[int, int, int, int]
        Bounding box B in format ``(x, y, width, height)``.

    Returns
    -------
    float
        Intersection-over-Union value in [0, 1].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou


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
