# background_reconstructor_mog2.py
from __future__ import annotations

import cv2
import numpy as np


class BackgroundReconstructorMog2:
    """Reconstruct a background using OpenCV's MOG2 model.

    MOG2 maintains an adaptive Gaussian-mixture background model for each
    pixel. The background image estimated by the internal model is retrieved
    after each frame using ``getBackgroundImage``.

    This reconstruction strategy operates independently of ViBe and does not
    use the externally supplied foreground mask, temporal reliability
    counters, or exponential moving averaging.

    The ``binary_mask`` argument is accepted only to preserve a common
    interface with the other reconstruction strategies.
    """

    def __init__(
        self,
        sample_frame: np.ndarray,
        background_reconstructor_settings: dict,
    ):
        """Initialize the MOG2 background reconstructor.

        Parameters
        ----------
        sample_frame : np.ndarray
            Initial three-channel frame with shape ``(H, W, 3)``.

        background_reconstructor_settings : dict
            Configuration dictionary.

            Supported MOG2-specific keys are:

            - ``"mog2_history"`` (int):
              Number of frames affecting the MOG2 background model.
              Default is ``500``.

            - ``"mog2_var_threshold"`` (float):
              Variance threshold used by MOG2 to determine whether a pixel
              matches the background model. Default is ``16.0``.

            - ``"mog2_detect_shadows"`` (bool):
              Whether MOG2 should perform shadow detection.
              Default is ``False``.

            - ``"mog2_learning_rate"`` (float):
              Learning rate passed to ``BackgroundSubtractorMOG2.apply``.
              A value of ``-1`` lets OpenCV choose the learning rate
              automatically. Default is ``-1.0``.

        Raises
        ------
        ValueError
            If the sample frame does not have shape ``(H, W, 3)``, if history
            is not positive, if the variance threshold is not positive, or if
            the learning rate lies outside the valid interval ``[-1, 1]``.
        """
        if sample_frame.ndim != 3 or sample_frame.shape[2] != 3:
            raise ValueError(
                "sample_frame must have shape (H, W, 3). " f"Got {sample_frame.shape}.",
            )

        self.__height: int = sample_frame.shape[0]
        self.__width: int = sample_frame.shape[1]

        self.__history: int = int(
            background_reconstructor_settings.get(
                "mog2_history",
                500,
            ),
        )

        self.__var_threshold: float = float(
            background_reconstructor_settings.get(
                "mog2_var_threshold",
                16.0,
            ),
        )

        self.__detect_shadows: bool = bool(
            background_reconstructor_settings.get(
                "mog2_detect_shadows",
                False,
            ),
        )

        self.__learning_rate: float = float(
            background_reconstructor_settings.get(
                "mog2_learning_rate",
                -1.0,
            ),
        )

        if self.__history <= 0:
            raise ValueError(
                "mog2_history must be > 0. " f"Got {self.__history}.",
            )

        if self.__var_threshold <= 0:
            raise ValueError(
                "mog2_var_threshold must be > 0. " f"Got {self.__var_threshold}.",
            )

        if not -1.0 <= self.__learning_rate <= 1.0:
            raise ValueError(
                "mog2_learning_rate must be in [-1, 1]. " f"Got {self.__learning_rate}.",
            )

        self.__mog2 = cv2.createBackgroundSubtractorMOG2(
            history=self.__history,
            varThreshold=self.__var_threshold,
            detectShadows=self.__detect_shadows,
        )

        self.__background: np.ndarray = sample_frame.astype(
            np.uint8,
            copy=True,
        )

        # Initialize the MOG2 model using the calibration frame.
        self.__mog2.apply(
            self.__background,
            learningRate=self.__learning_rate,
        )

        initial_background = self.__mog2.getBackgroundImage()

        if initial_background is not None:
            self.__background = initial_background.astype(
                np.uint8,
                copy=True,
            )

    def __call__(
        self,
        frame: np.ndarray,
        binary_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update the MOG2 model and return its background estimate.

        Parameters
        ----------
        frame : np.ndarray
            Current three-channel frame with shape ``(H, W, 3)``.

        binary_mask : np.ndarray | None
            Ignored. MOG2 performs its own foreground/background
            classification internally.

        Returns
        -------
        np.ndarray
            Current MOG2 background estimate with shape ``(H, W, 3)``
            and dtype ``uint8``.

        Raises
        ------
        ValueError
            If the current frame dimensions differ from the calibration
            frame dimensions.
        """
        del binary_mask

        if frame.shape != (
            self.__height,
            self.__width,
            3,
        ):
            raise ValueError(
                "frame shape must match sample_frame. " f"Expected " f"{(self.__height, self.__width, 3)}, " f"got {frame.shape}.",
            )

        frame_uint8: np.ndarray = frame.astype(
            np.uint8,
            copy=False,
        )

        # Update MOG2's internal Gaussian-mixture model.
        self.__mog2.apply(
            frame_uint8,
            learningRate=self.__learning_rate,
        )

        estimated_background = self.__mog2.getBackgroundImage()

        # OpenCV may return None before a usable background estimate
        # is available. In that case, retain the previous estimate.
        if estimated_background is not None:
            self.__background = estimated_background.astype(
                np.uint8,
                copy=True,
            )

        return self.__background.copy()
