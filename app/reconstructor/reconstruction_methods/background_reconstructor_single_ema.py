from __future__ import annotations

from typing import Final

import numpy as np


class BackgroundReconstructorSingleEma:
    """Reconstruct a background using static-pixel aging and a single EMA.

    The reconstructor tracks how long each pixel has continuously been
    classified as background. A pixel becomes eligible for background
    reconstruction after reaching a configurable age threshold. Reliable
    pixels are then ukpdated using a single exponential moving average (EMA)
    learning rate.

    Attributes
    ----------
    CHANNEL_COUNT : int
        Number of image channels. Fixed to 3 for RGB input.
    """

    CHANNEL_COUNT: Final[int] = 3

    def __init__(
        self,
        sample_frame: np.ndarray,
        background_reconstructor_settings: dict,
    ):
        """Initialize the background reconstructor.

        Parameters
        ----------
        sample_frame : np.ndarray
            Sample RGB frame used to determine the frame dimensions and
            initialize the reconstructed background. Expected shape is
            ``(H, W, 3)``.

        background_reconstructor_settings : dict
            Configuration dictionary containing the reconstruction parameters.

            Supported keys are:

            - ``"age_threshold"`` (int): Number of consecutive frames for
              which a pixel must be classified as background before becoming
              reliable. Default is ``20``.

            - ``"background_change_ratio"`` (float): EMA learning rate used
              for reliable background pixels. Default is ``0.02``.

        Raises
        ------
        ValueError
            If ``age_threshold`` is less than or equal to zero.

        ValueError
            If ``background_change_ratio`` is outside the interval ``(0, 1]``.
        """
        self.__height, self.__width, _ = sample_frame.shape

        self.__background: np.ndarray = sample_frame.astype(np.float32).copy()

        self.__static_pixel_duration_map: np.ndarray = np.zeros(
            (self.__height, self.__width),
            dtype=np.uint16,
        )

        self.__reliable_mask: np.ndarray | None = None

        self.__age_threshold: int = background_reconstructor_settings.get(
            "age_threshold",
            20,
        )

        self.__alpha: float = background_reconstructor_settings.get(
            "background_change_ratio",
            0.02,
        )

        if self.__age_threshold <= 0:
            raise ValueError(
                f"age_threshold must be > 0. " f"Got {self.__age_threshold}.",
            )

        if not 0.0 < self.__alpha <= 1.0:
            raise ValueError(
                "background_change_ratio must be in (0, 1]. " f"Got {self.__alpha}.",
            )

    def __call__(
        self,
        frame: np.ndarray,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """Process one frame and update the reconstructed background.

        Parameters
        ----------
        frame : np.ndarray
            Current RGB frame with shape ``(H, W, 3)``.

        binary_mask : np.ndarray
            Binary foreground mask with shape ``(H, W)``. A value of ``0``
            indicates background pixels, while nonzero values indicate
            foreground pixels.

        Returns
        -------
        np.ndarray
            Updated reconstructed background with shape ``(H, W, 3)`` and
            dtype ``uint8``.
        """
        self.__update_reliable_mask(binary_mask)

        reconstructed_background = self.__update_background(frame)

        return reconstructed_background.astype(np.uint8)

    def __update_reliable_mask(
        self,
        binary_mask: np.ndarray,
    ) -> None:
        """Update background-pixel ages and construct the reliability mask.

        Pixels classified as background increment their consecutive-background
        counter. Pixels classified as foreground have their counter reset to
        zero.

        A pixel becomes reliable once its counter reaches ``age_threshold``.
        The counter is saturated at the threshold because larger values are
        not required by the reconstruction algorithm.

        Parameters
        ----------
        binary_mask : np.ndarray
            Binary foreground mask with shape ``(H, W)``. Pixels equal to
            ``0`` are treated as background, while nonzero pixels are treated
            as foreground.
        """
        is_background: np.ndarray = binary_mask == 0

        # Foreground pixels lose their consecutive-background history.
        self.__static_pixel_duration_map[~is_background] = 0

        # Increment background-pixel ages up to the reliability threshold.
        increment_mask: np.ndarray = is_background & (self.__static_pixel_duration_map < self.__age_threshold)

        self.__static_pixel_duration_map[increment_mask] += 1

        self.__reliable_mask = self.__static_pixel_duration_map >= self.__age_threshold

    def __update_background(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Update the reconstructed background using a single EMA.

        Reliable pixels are updated using ``background_change_ratio``.
        Pixels that have not yet reached the reliability threshold remain
        unchanged.

        Parameters
        ----------
        frame : np.ndarray
            Current RGB frame with shape ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Updated reconstructed background with shape ``(H, W, 3)`` and
            dtype ``float32``.
        """
        reliable: np.ndarray = self.__reliable_mask

        alpha: float = self.__alpha

        for channel in range(self.CHANNEL_COUNT):
            background_channel: np.ndarray = self.__background[:, :, channel]

            current_channel: np.ndarray = frame[:, :, channel].astype(
                np.float32,
                copy=False,
            )

            background_channel[reliable] = (1.0 - alpha) * background_channel[reliable] + alpha * current_channel[reliable]

        return self.__background
