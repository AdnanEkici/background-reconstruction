# background_reconstructor_color_median.py
from __future__ import annotations

import os
from typing import Final

import cv2
import numpy as np


class BackgroundReconstructorColorMedian:
    """Reconstruct a background using a temporal color vector median.

    Each input frame is resized to a configurable working resolution before
    being stored. The default working resolution is 320 x 240.

    For each spatial pixel, the algorithm selects one observed color vector
    whose total Euclidean distance to all other temporal observations at that
    pixel is minimal:

        B(x) = argmin_{I_k(x)}
               sum_j ||I_k(x) - I_j(x)||_2

    Therefore, unlike an independent channel-wise median, the selected output
    color is always one of the actual color vectors observed at that pixel.

    This is a batch reconstruction method:

        frame 1 -> store
        frame 2 -> store
        ...
        frame T -> store
        finalize() -> compute final background

    The binary mask is ignored.
    """

    DEFAULT_WIDTH: Final[int] = 320
    DEFAULT_HEIGHT: Final[int] = 240

    DEFAULT_PIXEL_CHUNK_SIZE: Final[int] = 1024

    DEFAULT_OUTPUT_PATH: Final[str] = os.path.join(
        "output",
        "color_median_background_320x240.png",
    )

    def __init__(
        self,
        sample_frame: np.ndarray,
        background_reconstructor_settings: dict,
    ):
        """Initialize the Color Median reconstructor.

        Parameters
        ----------
        sample_frame : np.ndarray
            Initial input frame with shape (H, W, 3).

        background_reconstructor_settings : dict
            Configuration dictionary.

            Supported keys:

            color_median_width:
                Working image width.
                Default: 320.

            color_median_height:
                Working image height.
                Default: 240.

            color_median_pixel_chunk_size:
                Number of spatial pixels processed in one chunk during
                finalization.
                Default: 1024.

            color_median_output_path:
                Output path for the final reconstructed background.

        Raises
        ------
        ValueError
            If the input frame or configuration is invalid.
        """
        if sample_frame.ndim != 3 or sample_frame.shape[2] != 3:
            raise ValueError(
                "sample_frame must have shape (H, W, 3). " f"Got {sample_frame.shape}.",
            )

        self.__input_height: int = sample_frame.shape[0]
        self.__input_width: int = sample_frame.shape[1]

        self.__working_width: int = int(
            background_reconstructor_settings.get(
                "color_median_width",
                self.DEFAULT_WIDTH,
            ),
        )

        self.__working_height: int = int(
            background_reconstructor_settings.get(
                "color_median_height",
                self.DEFAULT_HEIGHT,
            ),
        )

        self.__pixel_chunk_size: int = int(
            background_reconstructor_settings.get(
                "color_median_pixel_chunk_size",
                self.DEFAULT_PIXEL_CHUNK_SIZE,
            ),
        )

        self.__output_path: str = str(
            background_reconstructor_settings.get(
                "color_median_output_path",
                self.DEFAULT_OUTPUT_PATH,
            ),
        )

        if self.__working_width <= 0:
            raise ValueError(
                "color_median_width must be > 0. " f"Got {self.__working_width}.",
            )

        if self.__working_height <= 0:
            raise ValueError(
                "color_median_height must be > 0. " f"Got {self.__working_height}.",
            )

        if self.__pixel_chunk_size <= 0:
            raise ValueError(
                "color_median_pixel_chunk_size must be > 0. " f"Got {self.__pixel_chunk_size}.",
            )

        if not self.__output_path.strip():
            raise ValueError(
                "color_median_output_path cannot be empty.",
            )

        resized_sample_frame: np.ndarray = self.__resize_frame(
            sample_frame,
        )

        self.__frames: list[np.ndarray] = [
            resized_sample_frame.copy(),
        ]

        self.__background: np.ndarray = resized_sample_frame.copy()

        self.__processed_frame_count: int = 0

        self.__finalized: bool = False

    def __call__(
        self,
        frame: np.ndarray,
        binary_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Store one resized temporal observation.

        Parameters
        ----------
        frame : np.ndarray
            Current frame with the same original shape as sample_frame.

        binary_mask : np.ndarray | None
            Ignored.

        Returns
        -------
        np.ndarray
            Current background preview at the Color Median working
            resolution.

            Until finalize() runs, this remains the resized initialization
            frame.

        Raises
        ------
        ValueError
            If the input frame does not match the expected dimensions.

        RuntimeError
            If called after finalization.
        """
        del binary_mask

        if self.__finalized:
            raise RuntimeError(
                "Color Median has already been finalized. " "No additional frames can be processed.",
            )

        if frame.shape != (
            self.__input_height,
            self.__input_width,
            3,
        ):
            raise ValueError(
                "frame shape must match sample_frame. " f"Expected " f"{(self.__input_height, self.__input_width, 3)}, " f"got {frame.shape}.",
            )

        resized_frame: np.ndarray = self.__resize_frame(
            frame,
        )

        self.__frames.append(
            resized_frame,
        )

        self.__processed_frame_count += 1

        return self.__background.copy()

    @property
    def output_path(self) -> str:
        """Return the configured output path."""
        return self.__output_path

    @property
    def working_width(self) -> int:
        """Return the Color Median working width."""
        return self.__working_width

    @property
    def working_height(self) -> int:
        """Return the Color Median working height."""
        return self.__working_height

    @property
    def pixel_chunk_size(self) -> int:
        """Return the configured spatial chunk size."""
        return self.__pixel_chunk_size

    @property
    def processed_frame_count(self) -> int:
        """Return the number of streamed frames collected."""
        return self.__processed_frame_count

    @property
    def finalized(self) -> bool:
        """Return whether finalization has completed."""
        return self.__finalized

    def finalize(self) -> np.ndarray:
        """Compute, save, and return the exact temporal color vector median.

        The computation is performed using spatial chunks to limit temporary
        memory consumption.

        Progress is printed while processing so long-running finalization does
        not appear to be frozen.

        Returns
        -------
        np.ndarray
            Final Color Median background with shape

                (color_median_height, color_median_width, 3)

            and dtype uint8.

        Raises
        ------
        RuntimeError
            If finalization has already occurred.

        IOError
            If the resulting image cannot be saved.
        """
        if self.__finalized:
            raise RuntimeError(
                "Color Median has already been finalized.",
            )

        frame_count: int = len(
            self.__frames,
        )

        if frame_count == 0:
            raise RuntimeError(
                "No frames are available for Color Median.",
            )

        print()
        print("=" * 60)
        print("COLOR MEDIAN FINALIZATION")
        print("=" * 60)
        print(
            f"Frames          : {frame_count}",
        )
        print(
            f"Resolution      : " f"{self.__working_width} x " f"{self.__working_height}",
        )
        print(
            f"Chunk size      : " f"{self.__pixel_chunk_size}",
        )
        print(
            "Computing exact temporal color vector median...",
        )
        print("=" * 60)

        # Shape:
        #
        # (T, H, W, 3)
        frames: np.ndarray = np.stack(
            self.__frames,
            axis=0,
        )

        # Shape:
        #
        # (T, P, 3)
        #
        # P = H * W
        pixels: np.ndarray = frames.reshape(
            frame_count,
            -1,
            3,
        ).astype(
            np.float32,
            copy=False,
        )

        pixel_count: int = pixels.shape[1]

        output: np.ndarray = np.empty(
            (
                pixel_count,
                3,
            ),
            dtype=np.uint8,
        )

        total_chunks: int = (pixel_count + self.__pixel_chunk_size - 1) // self.__pixel_chunk_size

        chunk_number: int = 0

        for start in range(
            0,
            pixel_count,
            self.__pixel_chunk_size,
        ):
            chunk_number += 1

            end: int = min(
                start + self.__pixel_chunk_size,
                pixel_count,
            )

            # Shape:
            #
            # (T, current_chunk_size, 3)
            chunk: np.ndarray = pixels[
                :,
                start:end,
                :,
            ]

            current_chunk_size: int = chunk.shape[1]

            best_cost: np.ndarray = np.full(
                current_chunk_size,
                np.inf,
                dtype=np.float32,
            )

            best_index: np.ndarray = np.zeros(
                current_chunk_size,
                dtype=np.int32,
            )

            for candidate_index in range(
                frame_count,
            ):
                # Shape:
                #
                # (1, current_chunk_size, 3)
                candidate: np.ndarray = chunk[
                    candidate_index : candidate_index + 1,
                    :,
                    :,
                ]

                # Shape:
                #
                # (T, current_chunk_size, 3)
                delta: np.ndarray = chunk - candidate

                # Euclidean color distance:
                #
                # (T, current_chunk_size)
                distances: np.ndarray = np.sqrt(
                    np.sum(
                        delta * delta,
                        axis=2,
                    ),
                )

                # Sum over temporal observations:
                #
                # (current_chunk_size,)
                cost: np.ndarray = np.sum(
                    distances,
                    axis=0,
                )

                better: np.ndarray = cost < best_cost

                best_cost[better] = cost[better]

                best_index[better] = candidate_index

            local_pixel_indices: np.ndarray = np.arange(
                current_chunk_size,
                dtype=np.int32,
            )

            selected: np.ndarray = chunk[
                best_index,
                local_pixel_indices,
                :,
            ]

            output[start:end] = selected.astype(
                np.uint8,
            )

            progress: float = chunk_number / total_chunks * 100.0

            print(
                f"\rColor Median progress: " f"{chunk_number}/{total_chunks} " f"({progress:6.2f}%)",
                end="",
                flush=True,
            )

        print()

        self.__background = output.reshape(
            self.__working_height,
            self.__working_width,
            3,
        )

        self.__finalized = True

        self.__save_background()

        # Free the stored video frames once the final
        # background has been produced.
        self.__frames.clear()

        print()
        print("=" * 60)
        print("COLOR MEDIAN FINALIZATION COMPLETE")
        print("=" * 60)

        return self.__background.copy()

    def __resize_frame(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Resize one frame to the configured Color Median resolution."""
        resized: np.ndarray = cv2.resize(
            frame,
            (
                self.__working_width,
                self.__working_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        return resized.astype(
            np.uint8,
            copy=False,
        )

    def __save_background(self) -> None:
        """Save the finalized Color Median background."""
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

        success: bool = cv2.imwrite(
            absolute_output_path,
            self.__background,
        )

        if not success:
            raise OSError(
                "Failed to save Color Median background to: " f"{absolute_output_path}",
            )

        print()
        print("=" * 60)
        print("COLOR MEDIAN BACKGROUND SAVED")
        print("=" * 60)
        print(
            f"Path       : {absolute_output_path}",
        )
        print(
            f"Resolution : " f"{self.__working_width} x " f"{self.__working_height}",
        )
        print(
            f"Shape      : " f"{self.__background.shape}",
        )
        print("=" * 60)
