# background_reconstructor_labgen.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np


@dataclass
class _SelectedPatch:
    """Store one LaBGen patch candidate."""

    motion_quantity: float
    frame_index: int
    pixels: np.ndarray


class BackgroundReconstructorLaBGen:
    """Reconstruct a background using LaBGen-style patch selection.

    The method operates as follows:

    1. A foreground/background mask is supplied by an external
       foreground-segmentation algorithm. In this project, ViBe is used.

    2. Each frame is divided into an N x N grid of spatial patches.

    3. For every spatial patch, the proportion of foreground pixels is used
       as a quantity-of-motion measure.

    4. For each spatial region, the S patches having the smallest quantities
       of motion are retained.

    5. At the end of the sequence, the retained patches are combined using
       a pixel-wise temporal median.

    6. The final reconstructed background is saved automatically to disk
       during ``finalize()``.

    The foreground-mask convention is:

        0       -> background
        nonzero -> foreground

    The default LaBGen parameters used here are:

        P = 1
        N = 2
        S = 19
    """

    DEFAULT_GRID_SIZE: Final[int] = 2
    DEFAULT_SELECTED_PATCH_COUNT: Final[int] = 19

    DEFAULT_OUTPUT_PATH: Final[str] = os.path.join(
        "output",
        "labgen_background.png",
    )

    PASSES: Final[int] = 1

    def __init__(
        self,
        sample_frame: np.ndarray,
        background_reconstructor_settings: dict,
    ):
        """Initialize the LaBGen reconstructor.

        Parameters
        ----------
        sample_frame : np.ndarray
            Calibration frame with shape ``(H, W, 3)``.

        background_reconstructor_settings : dict
            Configuration dictionary.

            Supported LaBGen-specific keys are:

            - ``"labgen_grid_size"`` (int):
              Number N of spatial divisions along each image dimension.
              Default is ``2``.

            - ``"labgen_selected_patch_count"`` (int):
              Number S of low-motion temporal patches retained for each
              spatial region.
              Default is ``19``.

            - ``"labgen_output_path"`` (str):
              File path used to save the final LaBGen reconstructed
              background.
              Default is ``"output/labgen_background.png"``.

        Raises
        ------
        ValueError
            If the sample frame or configuration is invalid.
        """
        if sample_frame.ndim != 3 or sample_frame.shape[2] != 3:
            raise ValueError(
                "sample_frame must have shape (H, W, 3). " f"Got {sample_frame.shape}.",
            )

        self.__height: int = sample_frame.shape[0]
        self.__width: int = sample_frame.shape[1]

        self.__grid_size: int = int(
            background_reconstructor_settings.get(
                "labgen_grid_size",
                self.DEFAULT_GRID_SIZE,
            ),
        )

        self.__selected_patch_count: int = int(
            background_reconstructor_settings.get(
                "labgen_selected_patch_count",
                self.DEFAULT_SELECTED_PATCH_COUNT,
            ),
        )

        self.__output_path: str = str(
            background_reconstructor_settings.get(
                "labgen_output_path",
                self.DEFAULT_OUTPUT_PATH,
            ),
        )

        if self.__grid_size <= 0:
            raise ValueError(
                "labgen_grid_size must be > 0. " f"Got {self.__grid_size}.",
            )

        if self.__grid_size > min(
            self.__height,
            self.__width,
        ):
            raise ValueError(
                "labgen_grid_size cannot be larger than "
                "the smallest image dimension. "
                f"Got grid size {self.__grid_size} for "
                f"image shape {sample_frame.shape}.",
            )

        if self.__selected_patch_count <= 0:
            raise ValueError(
                "labgen_selected_patch_count must be > 0. " f"Got {self.__selected_patch_count}.",
            )

        if self.__selected_patch_count % 2 == 0:
            raise ValueError(
                "labgen_selected_patch_count should be odd "
                "so that the temporal median corresponds to "
                "an actual middle observation. "
                f"Got {self.__selected_patch_count}.",
            )

        if not self.__output_path.strip():
            raise ValueError(
                "labgen_output_path cannot be empty.",
            )

        self.__background: np.ndarray = sample_frame.astype(
            np.uint8,
            copy=True,
        )

        self.__frame_index: int = 0

        self.__row_boundaries: np.ndarray = np.linspace(
            0,
            self.__height,
            self.__grid_size + 1,
            dtype=np.int32,
        )

        self.__column_boundaries: np.ndarray = np.linspace(
            0,
            self.__width,
            self.__grid_size + 1,
            dtype=np.int32,
        )

        patch_count: int = self.__grid_size * self.__grid_size

        self.__selected_patches: list[list[_SelectedPatch]] = [[] for _ in range(patch_count)]

        self.__finalized: bool = False

    def __call__(
        self,
        frame: np.ndarray,
        binary_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process one frame and update the low-motion patch subsets.

        Parameters
        ----------
        frame : np.ndarray
            Current frame with shape ``(H, W, 3)``.

        binary_mask : np.ndarray | None
            Foreground mask with shape ``(H, W)``.

            Pixels equal to ``0`` are treated as background.
            Nonzero pixels are treated as foreground.

        Returns
        -------
        np.ndarray
            Current reconstructed background.

            LaBGen is a batch method, so the actual final background is
            generated only when ``finalize()`` is called.

        Raises
        ------
        ValueError
            If the frame or binary mask has an invalid shape.

        RuntimeError
            If a frame is supplied after finalization.
        """
        if self.__finalized:
            raise RuntimeError(
                "LaBGen has already been finalized. " "No additional frames can be processed.",
            )

        if frame.shape != (
            self.__height,
            self.__width,
            3,
        ):
            raise ValueError(
                "frame shape must match sample_frame. " f"Expected " f"{(self.__height, self.__width, 3)}, " f"got {frame.shape}.",
            )

        if binary_mask is None:
            raise ValueError(
                "LaBGen requires a foreground binary mask.",
            )

        if binary_mask.shape != (
            self.__height,
            self.__width,
        ):
            raise ValueError(
                "binary_mask must have shape " f"{(self.__height, self.__width)}. " f"Got {binary_mask.shape}.",
            )

        self.__frame_index += 1

        frame_uint8: np.ndarray = frame.astype(
            np.uint8,
            copy=False,
        )

        foreground_mask: np.ndarray = binary_mask != 0

        patch_index: int = 0

        for row_index in range(
            self.__grid_size,
        ):
            row_start: int = int(
                self.__row_boundaries[row_index],
            )

            row_end: int = int(
                self.__row_boundaries[row_index + 1],
            )

            for column_index in range(
                self.__grid_size,
            ):
                column_start: int = int(
                    self.__column_boundaries[column_index],
                )

                column_end: int = int(
                    self.__column_boundaries[column_index + 1],
                )

                mask_patch: np.ndarray = foreground_mask[
                    row_start:row_end,
                    column_start:column_end,
                ]

                motion_quantity: float = float(
                    np.mean(mask_patch),
                )

                frame_patch: np.ndarray = frame_uint8[
                    row_start:row_end,
                    column_start:column_end,
                    :,
                ].copy()

                self.__consider_patch(
                    patch_index=patch_index,
                    motion_quantity=motion_quantity,
                    patch=frame_patch,
                )

                patch_index += 1

        return self.__background.copy()

    @property
    def frame_count(self) -> int:
        """Return the number of processed candidate frames."""
        return self.__frame_index

    @property
    def grid_size(self) -> int:
        """Return LaBGen parameter N."""
        return self.__grid_size

    @property
    def selected_patch_count(self) -> int:
        """Return LaBGen parameter S."""
        return self.__selected_patch_count

    @property
    def output_path(self) -> str:
        """Return the configured output path."""
        return self.__output_path

    def finalize(self) -> np.ndarray:
        """Generate and save the final LaBGen background.

        The selected low-motion temporal patches for each spatial region are
        stacked and combined using a pixel-wise temporal median.

        The resulting image is automatically saved to
        ``labgen_output_path``.

        Returns
        -------
        np.ndarray
            Final reconstructed background with shape ``(H, W, 3)``
            and dtype ``uint8``.

        Raises
        ------
        RuntimeError
            If no frames were processed.

        IOError
            If OpenCV fails to save the reconstructed background.
        """
        if self.__frame_index == 0:
            raise RuntimeError(
                "Cannot finalize LaBGen because " "no frames were processed.",
            )

        reconstructed_background = np.empty(
            (
                self.__height,
                self.__width,
                3,
            ),
            dtype=np.uint8,
        )

        patch_index: int = 0

        for row_index in range(
            self.__grid_size,
        ):
            row_start: int = int(
                self.__row_boundaries[row_index],
            )

            row_end: int = int(
                self.__row_boundaries[row_index + 1],
            )

            for column_index in range(
                self.__grid_size,
            ):
                column_start: int = int(
                    self.__column_boundaries[column_index],
                )

                column_end: int = int(
                    self.__column_boundaries[column_index + 1],
                )

                selected: list[_SelectedPatch] = self.__selected_patches[patch_index]

                if not selected:
                    reconstructed_background[
                        row_start:row_end,
                        column_start:column_end,
                        :,
                    ] = self.__background[
                        row_start:row_end,
                        column_start:column_end,
                        :,
                    ]

                    patch_index += 1
                    continue

                temporal_patch_stack: np.ndarray = np.stack(
                    [item.pixels for item in selected],
                    axis=0,
                )

                median_patch: np.ndarray = np.median(
                    temporal_patch_stack,
                    axis=0,
                )

                reconstructed_background[
                    row_start:row_end,
                    column_start:column_end,
                    :,
                ] = np.clip(
                    median_patch,
                    0,
                    255,
                ).astype(np.uint8)

                patch_index += 1

        self.__background = reconstructed_background

        self.__finalized = True

        self.__save_background()

        return self.__background.copy()

    def __consider_patch(
        self,
        patch_index: int,
        motion_quantity: float,
        patch: np.ndarray,
    ) -> None:
        """Maintain the S lowest-motion patches for one spatial region.

        Parameters
        ----------
        patch_index : int
            Spatial-patch index.

        motion_quantity : float
            Fraction of foreground pixels in the patch.

        patch : np.ndarray
            Current color image patch.

        Returns
        -------
        None
        """
        selected: list[_SelectedPatch] = self.__selected_patches[patch_index]

        candidate = _SelectedPatch(
            motion_quantity=motion_quantity,
            frame_index=self.__frame_index,
            pixels=patch,
        )

        if len(selected) < self.__selected_patch_count:
            selected.append(candidate)
            return

        maximum_motion: float = max(item.motion_quantity for item in selected)

        if motion_quantity > maximum_motion:
            return

        removable_indices: list[int] = [
            index
            for index, item in enumerate(
                selected,
            )
            if (item.motion_quantity == maximum_motion)
        ]

        oldest_index: int = min(
            removable_indices,
            key=lambda index: (selected[index].frame_index),
        )

        selected[oldest_index] = candidate

    def __save_background(self) -> None:
        """Save the finalized LaBGen background to disk.

        The parent directory is created automatically when necessary.

        Raises
        ------
        IOError
            If OpenCV fails to write the image.
        """
        absolute_output_path: str = os.path.abspath(
            self.__output_path,
        )

        output_directory: str = os.path.dirname(
            absolute_output_path,
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True,
            )

        save_successful: bool = cv2.imwrite(
            absolute_output_path,
            self.__background,
        )

        if not save_successful:
            raise OSError(
                "Failed to save LaBGen reconstructed " f"background to: " f"{absolute_output_path}",
            )

        print()
        print("=" * 50)
        print("LABGEN BACKGROUND SAVED")
        print("=" * 50)
        print(
            f"Path   : {absolute_output_path}",
        )
        print(
            f"Shape  : {self.__background.shape}",
        )
        print(
            f"Frames : {self.__frame_index}",
        )
        print("=" * 50)
