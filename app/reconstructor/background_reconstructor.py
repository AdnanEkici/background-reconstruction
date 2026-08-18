from __future__ import annotations

from typing import Final

import numpy as np


class BackgroundReconstructor:
    """Reconstruct a clean background from a video stream using temporally stable pixels.

    This class tracks how long each pixel remains static and updates the background image
    using a dual-rate exponential moving average (EMA). Pixels that remain static for a
    configured threshold are considered reliable and contribute to background reconstruction.

    Attributes
    ----------
    CHANNEL_COUNT : int
        Number of image channels. Default is ``3`` (RGB).
    __height : int
        Frame height in pixels, derived from the sample frame.
    __width : int
        Frame width in pixels, derived from the sample frame.
    __background : np.ndarray
        Float32 background image updated incrementally.
    __static_pixel_duration_map : np.ndarray
        Map of static durations per pixel (uint8).
    __static_mask : np.ndarray or None
        Current reliability mask expanded to three channels, or ``None`` if unset.
    __age_threshold : int
        Number of frames a pixel must remain static to be considered reliable.
    __background_change_ratio : float
        EMA factor applied for long-term static pixels.
    __foreground_change_ratio : float
        EMA factor applied for newly static pixels.
    """

    CHANNEL_COUNT: Final[int] = 3

    def __init__(self, sample_frame: np.ndarray, background_reconstructor_settings: dict):
        """Initialize the background reconstructor.

        Parameters
        ----------
        sample_frame : np.ndarray
            Initial RGB frame used to set dimensions and initialize background storage.
            Shape ``(H, W, 3)``.
        background_reconstructor_settings : dict
            Configuration dictionary with keys:

            - ``"age_threshold"`` (int): Number of frames a pixel must remain static
              to be considered reliable. Default is 5.
            - ``"background_change_ratio"`` (float): EMA factor for long-term static pixels.
              Default is 0.02.
            - ``"foreground_change_ratio"`` (float): EMA factor for newly static pixels.
              Default is 0.01.
        """

        self.__height: int
        self.__width: int
        self.__height, self.__width, _ = sample_frame.shape
        self.__background: np.ndarray = np.zeros_like(sample_frame, dtype=np.float32)
        self.__static_pixel_duration_map: np.ndarray = np.zeros((self.__height, self.__width), dtype=np.uint8)

        self.__static_mask: np.ndarray | None = None

        self.__age_threshold: int = background_reconstructor_settings.get("age_threshold", 5)
        self.__background_change_ratio: float = background_reconstructor_settings.get("background_change_ratio", 0.02)
        self.__foreground_change_ratio: float = background_reconstructor_settings.get("foreground_change_ratio", 0.01)

    def __call__(self, frame: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        """Process a frame and update the background image based on static pixel analysis.

        Parameters
        ----------
        frame : np.ndarray
            The current RGB frame. Shape ``(H, W, 3)``.
        binary_mask : np.ndarray
            Binary foreground mask where 0 = background and 255 = foreground. Shape ``(H, W)``.

        Returns
        -------
        np.ndarray
            The updated background image as a uint8 array. Shape ``(H, W, 3)``.
        """
        just_became_static = self.__get_reliable_static_mask(binary_mask)
        reconstructed_background = self.__update_background(frame, just_became_static)
        return reconstructed_background.astype(np.uint8)

    def __get_reliable_static_mask(
        self,
        binary_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Identify pixels that have remained static for the threshold duration.

        Parameters
        ----------
        binary_mask : np.ndarray
            Foreground mask indicating motion (255 = motion, 0 = static). Shape ``(H, W)``.

        Returns
        -------
        np.ndarray
            Boolean mask where ``True`` indicates pixels that just became reliably static.
            Shape ``(H, W)``.
        """

        is_static: np.ndarray = binary_mask == 0
        pixels_just_became_reliable: np.ndarray = self.__static_pixel_duration_map == self.__age_threshold

        self.__static_pixel_duration_map[is_static] += 1
        self.__static_pixel_duration_map[~is_static] = 0

        reliable_static: np.ndarray = (self.__static_pixel_duration_map >= self.__age_threshold).astype(np.float32)
        self.__static_mask = np.repeat(reliable_static[:, :, None], self.CHANNEL_COUNT, axis=2)
        pixels_just_became_reliable = self.__static_pixel_duration_map == self.__age_threshold
        return pixels_just_became_reliable

    def __update_background(self, frame: np.ndarray, just_became_static: np.ndarray) -> np.ndarray:
        """Update the background image using EMA where reliable pixels are detected.

        Parameters
        ----------
        frame : np.ndarray
            The current RGB frame. Shape ``(H, W, 3)``.
        just_became_static : np.ndarray
            Boolean mask of pixels that just became reliably static. Shape ``(H, W)``.

        Returns
        -------
        np.ndarray
            Updated background image in float32 format. Shape ``(H, W, 3)``.
        """
        reliable: np.ndarray = self.__static_mask[:, :, 0] == 1
        alpha_map: np.ndarray = np.full((self.__height, self.__width), self.__background_change_ratio, dtype=np.float32)
        alpha_map[just_became_static] = self.__foreground_change_ratio

        for channel in range(self.CHANNEL_COUNT):
            single_channel_background: np.ndarray = self.__background[:, :, channel]
            current: np.ndarray = frame[:, :, channel].astype(np.float32)
            single_channel_background[reliable] = (1 - alpha_map[reliable]) * single_channel_background[
                reliable
            ] + alpha_map[reliable] * current[reliable]

        return self.__background
