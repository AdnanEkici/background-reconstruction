from __future__ import annotations

from typing import Final

import numpy as np


class BackgroundReconstructorDualEma:
    """Reconstruct a background using static-pixel aging and dual-rate EMA.

    The reconstructor tracks how long each pixel has continuously been
    classified as background. Pixels become eligible for reconstruction after
    reaching a configurable age threshold. Newly reliable pixels are updated
    using a lower exponential moving average (EMA) learning rate, while
    established reliable pixels are updated using a higher learning rate.

    Attributes
    ----------
    CHANNEL_COUNT : int
        Number of image channels. Fixed to 3 for RGB input.
    TRANSITION_DURATION : int
        Number of frames for which a newly reliable pixel uses the lower EMA
        learning rate before switching to the established learning rate.
    """

    CHANNEL_COUNT: Final[int] = 3
    TRANSITION_DURATION: Final[int] = 1

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
            - ``"foreground_change_ratio"`` (float): EMA learning rate used
              for newly reliable pixels. Default is ``0.01``.
            - ``"background_change_ratio"`` (float): EMA learning rate used
              for established reliable pixels. Default is ``0.02``.

        Raises
        ------
        ValueError
            If ``age_threshold`` is less than or equal to zero.
        ValueError
            If ``foreground_change_ratio`` is outside the interval ``(0, 1]``.
        ValueError
            If ``background_change_ratio`` is outside the interval ``(0, 1]``.
        ValueError
            If ``foreground_change_ratio`` is greater than or equal to
            ``background_change_ratio``.
        """
        self.__height, self.__width, _ = sample_frame.shape

        self.__background: np.ndarray = sample_frame.astype(np.float32).copy()

        self.__static_pixel_duration_map: np.ndarray = np.zeros(
            (self.__height, self.__width),
            dtype=np.uint16,
        )

        self.__newly_reliable_mask: np.ndarray | None = None
        self.__established_mask: np.ndarray | None = None

        self.__age_threshold: int = background_reconstructor_settings.get(
            "age_threshold",
            20,
        )

        self.__alpha_new: float = background_reconstructor_settings.get(
            "foreground_change_ratio",
            0.01,
        )

        self.__alpha_established: float = background_reconstructor_settings.get(
            "background_change_ratio",
            0.02,
        )

        if self.__age_threshold <= 0:
            raise ValueError(
                f"age_threshold must be > 0. Got {self.__age_threshold}.",
            )

        if not 0.0 < self.__alpha_new <= 1.0:
            raise ValueError(
                f"foreground_change_ratio must be in (0, 1]. " f"Got {self.__alpha_new}.",
            )

        if not 0.0 < self.__alpha_established <= 1.0:
            raise ValueError(
                f"background_change_ratio must be in (0, 1]. " f"Got {self.__alpha_established}.",
            )

        if self.__alpha_new >= self.__alpha_established:
            raise ValueError(
                "foreground_change_ratio must be smaller than "
                "background_change_ratio. "
                f"Got foreground_change_ratio={self.__alpha_new}, "
                f"background_change_ratio={self.__alpha_established}.",
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
            indicates background/static pixels, while nonzero values indicate
            foreground pixels.

        Returns
        -------
        np.ndarray
            Updated reconstructed background with shape ``(H, W, 3)`` and
            dtype ``uint8``.
        """

        self.__update_reliable_masks(binary_mask)

        reconstructed_background = self.__update_background(frame)

        return reconstructed_background.astype(np.uint8)

    def __update_reliable_masks(
        self,
        binary_mask: np.ndarray,
    ) -> None:
        """Update static-pixel ages and construct reliability masks.

        Pixels classified as background increment their static-duration
        counter, while foreground pixels have their counters reset to zero.
        The counter is saturated at the maximum age required by the dual-rate
        update scheme.

        Pixels whose age is between ``age_threshold`` and
        ``age_threshold + TRANSITION_DURATION`` are considered newly reliable
        and use the lower EMA learning rate. Pixels whose age reaches the
        maximum age are considered established and use the higher EMA
        learning rate.

        Parameters
        ----------
        binary_mask : np.ndarray
            Binary foreground mask with shape ``(H, W)``. Pixels equal to
            ``0`` are treated as background/static, while nonzero pixels are
            treated as foreground.
        """

        is_static: np.ndarray = binary_mask == 0

        # Foreground pixels lose their static history.
        self.__static_pixel_duration_map[~is_static] = 0

        maximum_age: int = self.__age_threshold + self.TRANSITION_DURATION

        # Saturate at maximum_age to avoid counter overflow.
        increment_mask: np.ndarray = is_static & (self.__static_pixel_duration_map < maximum_age)

        self.__static_pixel_duration_map[increment_mask] += 1

        age: np.ndarray = self.__static_pixel_duration_map

        # Slow EMA:
        self.__newly_reliable_mask = (age >= self.__age_threshold) & (age < maximum_age)

        # Faster EMA after the transition period.
        self.__established_mask = age >= maximum_age

    def __update_background(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Update the reconstructed background using dual-rate EMA.

        Newly reliable pixels are updated using ``foreground_change_ratio``,
        while established reliable pixels are updated using
        ``background_change_ratio``. Pixels that are not yet reliable are left
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

        newly_reliable: np.ndarray = self.__newly_reliable_mask
        established: np.ndarray = self.__established_mask

        alpha_new: float = self.__alpha_new
        alpha_established: float = self.__alpha_established

        for channel in range(self.CHANNEL_COUNT):
            background_channel: np.ndarray = self.__background[:, :, channel]

            current_channel: np.ndarray = frame[:, :, channel].astype(
                np.float32,
                copy=False,
            )

            background_channel[newly_reliable] = (1.0 - alpha_new) * background_channel[newly_reliable] + alpha_new * current_channel[newly_reliable]

            background_channel[established] = (1.0 - alpha_established) * background_channel[established] + alpha_established * current_channel[
                established
            ]

        return self.__background
