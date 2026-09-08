from __future__ import annotations

import numpy as np


class BackgroundReconstructorMean:
    """Reconstruct the background using a temporal arithmetic mean.

    The reconstructed background is the running pixel-wise mean of all
    processed frames. No foreground segmentation, temporal reliability
    threshold, or exponential moving average is used.

    The class accepts ``binary_mask`` only to maintain the same callable
    interface as the other reconstruction methods.
    """

    def __init__(
        self,
        sample_frame: np.ndarray,
        background_reconstructor_settings: dict,
    ):
        """Initialize the temporal-mean background reconstructor.

        Parameters
        ----------
        sample_frame : np.ndarray
            Initial frame with shape ``(H, W, 3)``. The sample frame is
            included as the first observation in the running mean.

        background_reconstructor_settings : dict
            Unused for this reconstruction strategy. It is accepted to
            preserve a common interface with the other methods.
        """
        del background_reconstructor_settings

        if sample_frame.ndim != 3 or sample_frame.shape[2] != 3:
            raise ValueError(
                "sample_frame must have shape (H, W, 3). " f"Got {sample_frame.shape}.",
            )

        self.__background: np.ndarray = sample_frame.astype(np.float32).copy()

        # The calibration/sample frame is the first observation.
        self.__frame_count: int = 1

    def __call__(
        self,
        frame: np.ndarray,
        binary_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update and return the temporal mean background.

        Parameters
        ----------
        frame : np.ndarray
            Current frame with shape ``(H, W, 3)``.

        binary_mask : np.ndarray | None
            Ignored. Included only for interface compatibility.

        Returns
        -------
        np.ndarray
            Running temporal-mean background with dtype ``uint8``.
        """
        del binary_mask

        self.__frame_count += 1

        frame_float: np.ndarray = frame.astype(
            np.float32,
            copy=False,
        )

        self.__background += (frame_float - self.__background) / self.__frame_count

        return np.clip(
            self.__background,
            0,
            255,
        ).astype(np.uint8)
