from __future__ import annotations

from typing import Final

import numpy as np


class BackgroundReconstructorNoEma:
    """Reconstruct a background using temporal reliability without EMA.

    The reconstructor tracks the number of consecutive frames for which each
    pixel has been classified as background. A pixel becomes reliable after
    reaching a configurable age threshold. Reliable pixels are copied directly
    from the current frame into the reconstructed background without applying
    exponential moving averaging or any other temporal smoothing.

    Attributes
    ----------
    CHANNEL_COUNT : int
        Number of image channels. Fixed to 3 for three-channel input.
    """

    CHANNEL_COUNT: Final[int] = 3

    def __init__(
        self,
        sample_frame: np.ndarray,
        background_reconstructor_settings: dict,
    ):
        """Initialize the no-EMA background reconstructor.

        Parameters
        ----------
        sample_frame : np.ndarray
            Sample frame used to determine the dimensions of the reconstructed
            background. Expected shape is ``(H, W, 3)``.

        background_reconstructor_settings : dict
            Configuration dictionary containing the reconstruction parameters.

            Supported keys are:

            - ``"age_threshold"`` (int): Number of consecutive frames for
              which a pixel must be classified as background before becoming
              reliable. Default is ``20``.

        Raises
        ------
        ValueError
            If ``age_threshold`` is less than or equal to zero.
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

        if self.__age_threshold <= 0:
            raise ValueError(
                f"age_threshold must be > 0. " f"Got {self.__age_threshold}.",
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
            Current three-channel frame with shape ``(H, W, 3)``.

        binary_mask : np.ndarray
            Binary foreground mask with shape ``(H, W)``. Pixels equal to
            ``0`` are treated as background, while nonzero pixels are treated
            as foreground.

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
        """Update consecutive-background ages and the reliability mask.

        Pixels classified as background increment their consecutive-background
        counter. Pixels classified as foreground have their counter reset to
        zero.

        A pixel becomes reliable once its counter reaches
        ``age_threshold``. The counter is saturated at this threshold because
        additional age information is not required by the direct-replacement
        reconstruction strategy.

        Parameters
        ----------
        binary_mask : np.ndarray
            Binary foreground mask with shape ``(H, W)``. Pixels equal to
            ``0`` are treated as background, while nonzero pixels are treated
            as foreground.

        Returns
        -------
        None
        """
        is_static: np.ndarray = binary_mask == 0

        # Foreground pixels lose their consecutive-background history.
        self.__static_pixel_duration_map[~is_static] = 0

        # Increment background-pixel ages up to the reliability threshold.
        increment_mask: np.ndarray = is_static & (self.__static_pixel_duration_map < self.__age_threshold)

        self.__static_pixel_duration_map[increment_mask] += 1

        self.__reliable_mask = self.__static_pixel_duration_map >= self.__age_threshold

    def __update_background(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Update reliable background pixels by direct replacement.

        Pixels that have reached the reliability threshold are copied directly
        from the current frame into the reconstructed background. Pixels that
        are not yet reliable retain their previous reconstructed values.

        No exponential moving average is applied.

        Parameters
        ----------
        frame : np.ndarray
            Current three-channel frame with shape ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Updated reconstructed background with shape ``(H, W, 3)`` and
            dtype ``float32``.
        """
        reliable: np.ndarray = self.__reliable_mask

        # Keep the same channel-by-channel memory access pattern as the EMA
        # variants for a comparable runtime implementation.
        for channel in range(self.CHANNEL_COUNT):
            background_channel: np.ndarray = self.__background[:, :, channel]

            current_channel: np.ndarray = frame[:, :, channel]

            background_channel[reliable] = current_channel[reliable]

        return self.__background
