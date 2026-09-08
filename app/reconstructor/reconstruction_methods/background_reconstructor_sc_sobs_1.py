# background_reconstructor_sc_sobs_1.py
from __future__ import annotations

import os
from typing import Final

import cv2
import numpy as np


class BackgroundReconstructorScSobs1:
    """SC-SOBS_1 background-initialization baseline.

    This class implements the principal SC-SOBS neural-background-model
    operations described by Maddalena and Petrosino:

    - n x n neuronal weight vectors per image pixel;
    - initialization from the first frame;
    - HSV-hexcone color representation;
    - best-matching weight-vector selection;
    - spatial Neighborhood Coherence Factor (NCF);
    - selective self-organizing update;
    - training/testing learning-rate schedules.

    IMPORTANT
    ---------
    The SBI SC-SOBS_1 background estimate is an oracle-style result.

    After the SC-SOBS model has been learned, the final background pixel is
    selected from the n^2 learned weight vectors by choosing the weight vector
    closest to the provided ground-truth background pixel.

    Consequently, SC-SOBS_1 requires the ground-truth background image during
    finalize().

    It must not be described as an unsupervised/deployable online baseline.

    Mask convention used internally:

        0   -> background
        255 -> foreground

    This implementation does not use the externally supplied ViBe mask.
    """

    DEFAULT_NEURON_GRID_SIZE: Final[int] = 3

    DEFAULT_TRAINING_THRESHOLD: Final[float] = 1.0
    DEFAULT_TESTING_THRESHOLD: Final[float] = 0.008

    DEFAULT_TRAINING_LEARNING_RATE: Final[float] = 1.0
    DEFAULT_TESTING_LEARNING_RATE: Final[float] = 0.05

    DEFAULT_COHERENCE_WINDOW_SIZE: Final[int] = 5

    DEFAULT_NEURAL_UPDATE_RADIUS: Final[int] = 1
    DEFAULT_GAUSSIAN_SIGMA: Final[float] = 1.0

    DEFAULT_TRAINING_FRAMES: Final[int] = 500

    DEFAULT_OUTPUT_PATH: Final[str] = os.path.join(
        "output",
        "sc_sobs_1_background.png",
    )

    def __init__(
        self,
        sample_frame: np.ndarray,
        background_reconstructor_settings: dict,
    ):
        if sample_frame.ndim != 3 or sample_frame.shape[2] != 3:
            raise ValueError(
                "sample_frame must have shape (H, W, 3). " f"Got {sample_frame.shape}.",
            )

        self.__height: int = sample_frame.shape[0]
        self.__width: int = sample_frame.shape[1]

        self.__neuron_grid_size: int = int(
            background_reconstructor_settings.get(
                "sc_sobs_neuron_grid_size",
                self.DEFAULT_NEURON_GRID_SIZE,
            ),
        )

        self.__training_threshold: float = float(
            background_reconstructor_settings.get(
                "sc_sobs_training_threshold",
                self.DEFAULT_TRAINING_THRESHOLD,
            ),
        )

        self.__testing_threshold: float = float(
            background_reconstructor_settings.get(
                "sc_sobs_testing_threshold",
                self.DEFAULT_TESTING_THRESHOLD,
            ),
        )

        self.__training_learning_rate: float = float(
            background_reconstructor_settings.get(
                "sc_sobs_training_learning_rate",
                self.DEFAULT_TRAINING_LEARNING_RATE,
            ),
        )

        self.__testing_learning_rate: float = float(
            background_reconstructor_settings.get(
                "sc_sobs_testing_learning_rate",
                self.DEFAULT_TESTING_LEARNING_RATE,
            ),
        )

        self.__coherence_window_size: int = int(
            background_reconstructor_settings.get(
                "sc_sobs_coherence_window_size",
                self.DEFAULT_COHERENCE_WINDOW_SIZE,
            ),
        )

        self.__update_radius: int = int(
            background_reconstructor_settings.get(
                "sc_sobs_update_radius",
                self.DEFAULT_NEURAL_UPDATE_RADIUS,
            ),
        )

        self.__gaussian_sigma: float = float(
            background_reconstructor_settings.get(
                "sc_sobs_gaussian_sigma",
                self.DEFAULT_GAUSSIAN_SIGMA,
            ),
        )

        self.__training_frames: int = int(
            background_reconstructor_settings.get(
                "sc_sobs_training_frames",
                self.DEFAULT_TRAINING_FRAMES,
            ),
        )

        self.__ground_truth_path: str = str(
            background_reconstructor_settings.get(
                "sc_sobs_ground_truth_path",
                "",
            ),
        )

        self.__output_path: str = str(
            background_reconstructor_settings.get(
                "sc_sobs_output_path",
                self.DEFAULT_OUTPUT_PATH,
            ),
        )

        self.__validate_configuration()

        # -------------------------------------------------------------
        # Convert the calibration frame into the HSV-hexcone embedding.
        #
        # Each color is represented as:
        #
        #   (v*s*cos(h), v*s*sin(h), v)
        #
        # -------------------------------------------------------------
        sample_hexcone: np.ndarray = self.__bgr_to_hexcone(
            sample_frame,
        )

        n: int = self.__neuron_grid_size

        # -------------------------------------------------------------
        # SC-SOBS neuronal map B.
        #
        # Input:
        #     H x W pixels
        #
        # Neuronal map:
        #     (H*n) x (W*n)
        #
        # Every image pixel therefore owns an n x n block of neurons.
        # All neurons are initialized from the first image.
        # -------------------------------------------------------------
        self.__neural_map: np.ndarray = np.repeat(
            np.repeat(
                sample_hexcone,
                n,
                axis=0,
            ),
            n,
            axis=1,
        ).astype(
            np.float32,
            copy=False,
        )

        self.__background: np.ndarray = sample_frame.astype(
            np.uint8,
            copy=True,
        )

        self.__frame_index: int = 0

        self.__finalized: bool = False

        self.__last_detection_mask: np.ndarray = np.zeros(
            (
                self.__height,
                self.__width,
            ),
            dtype=np.uint8,
        )

        self.__row_indices: np.ndarray = np.arange(
            self.__height,
            dtype=np.int32,
        )[:, None]

        self.__column_indices: np.ndarray = np.arange(
            self.__width,
            dtype=np.int32,
        )[None, :]

        self.__gaussian_weights: dict[
            tuple[int, int],
            float,
        ] = self.__build_gaussian_weights()

    def __call__(
        self,
        frame: np.ndarray,
        binary_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process one SC-SOBS frame.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame, shape (H, W, 3).

        binary_mask : np.ndarray | None
            Ignored. SC-SOBS performs its own background/foreground
            classification.

        Returns
        -------
        np.ndarray
            Current practical preview of the SC-SOBS model.

            The official SC-SOBS_1 oracle background is produced only by
            finalize(), because final selection requires the ground truth.
        """
        del binary_mask

        if self.__finalized:
            raise RuntimeError(
                "SC-SOBS_1 has already been finalized. " "No additional frames can be processed.",
            )

        if frame.shape != (
            self.__height,
            self.__width,
            3,
        ):
            raise ValueError(
                "frame shape must match sample_frame. " f"Expected " f"{(self.__height, self.__width, 3)}, " f"got {frame.shape}.",
            )

        self.__frame_index += 1

        current: np.ndarray = self.__bgr_to_hexcone(
            frame,
        )

        (
            best_local_row,
            best_local_column,
            best_distance,
        ) = self.__find_best_matches(
            current,
        )

        threshold: float = self.__current_threshold()

        # -------------------------------------------------------------
        # Omega condition:
        #
        # A pixel is represented by the model if its best neural vector
        # lies within epsilon.
        # -------------------------------------------------------------
        represented: np.ndarray = best_distance <= threshold

        # -------------------------------------------------------------
        # SC-SOBS Neighborhood Coherence Factor.
        #
        # NCF(p) = |Omega_p| / |N_p|
        #
        # OpenCV's normalized box filter directly gives the fraction of
        # represented pixels in the coherence neighborhood.
        # -------------------------------------------------------------
        ncf: np.ndarray = cv2.boxFilter(
            represented.astype(np.float32),
            ddepth=-1,
            ksize=(
                self.__coherence_window_size,
                self.__coherence_window_size,
            ),
            normalize=True,
            borderType=cv2.BORDER_REPLICATE,
        )

        # Paper definition:
        #
        # D(p) = 1 if NCF <= 0.5
        #        0 otherwise
        #
        foreground: np.ndarray = ncf <= 0.5

        background: np.ndarray = ~foreground

        self.__last_detection_mask = foreground.astype(np.uint8) * 255

        learning_rate: float = self.__current_learning_rate()

        self.__update_neural_map(
            current=current,
            best_local_row=best_local_row,
            best_local_column=best_local_column,
            background_mask=background,
            learning_rate=learning_rate,
        )

        # Produce only a practical visualization while training.
        #
        # This is NOT the SBI SC-SOBS_1 oracle result.
        self.__background = self.__extract_central_model()

        return self.__background.copy()

    @property
    def detection_mask(self) -> np.ndarray:
        """Return the most recent SC-SOBS foreground mask."""
        return self.__last_detection_mask.copy()

    @property
    def frame_count(self) -> int:
        """Return number of processed frames."""
        return self.__frame_index

    @property
    def finalized(self) -> bool:
        """Return whether finalize() has been called."""
        return self.__finalized

    @property
    def output_path(self) -> str:
        """Return configured output path."""
        return self.__output_path

    def finalize(self) -> np.ndarray:
        """Generate the SBI-style SC-SOBS_1 oracle background.

        For each image pixel, the n^2 learned SC-SOBS weight vectors are
        compared against that pixel in the supplied ground-truth background.

        The closest learned weight vector is selected.

        This follows the evaluation-only extraction strategy described for
        SC-SOBS in the SBI benchmarking work.

        Returns
        -------
        np.ndarray
            Final SC-SOBS_1 background image.

        Raises
        ------
        RuntimeError
            If finalization has already occurred.

        FileNotFoundError
            If the configured ground-truth image cannot be read.
        """
        if self.__finalized:
            raise RuntimeError(
                "SC-SOBS_1 has already been finalized.",
            )

        ground_truth: np.ndarray | None = cv2.imread(
            self.__ground_truth_path,
            cv2.IMREAD_COLOR,
        )

        if ground_truth is None:
            raise FileNotFoundError(
                "Could not load SC-SOBS_1 ground truth from: " f"{self.__ground_truth_path}",
            )

        if ground_truth.shape[:2] != (
            self.__height,
            self.__width,
        ):
            ground_truth = cv2.resize(
                ground_truth,
                (
                    self.__width,
                    self.__height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        ground_truth_hexcone: np.ndarray = self.__bgr_to_hexcone(
            ground_truth,
        )

        n: int = self.__neuron_grid_size

        models: np.ndarray = self.__neural_map.reshape(
            self.__height,
            n,
            self.__width,
            n,
            3,
        ).transpose(
            0,
            2,
            1,
            3,
            4,
        )

        difference: np.ndarray = (
            models
            - ground_truth_hexcone[
                :,
                :,
                None,
                None,
                :,
            ]
        )

        distance_squared: np.ndarray = np.sum(
            difference * difference,
            axis=-1,
        )

        flat_distance: np.ndarray = distance_squared.reshape(
            self.__height,
            self.__width,
            n * n,
        )

        best_index: np.ndarray = np.argmin(
            flat_distance,
            axis=2,
        )

        flat_models: np.ndarray = models.reshape(
            self.__height,
            self.__width,
            n * n,
            3,
        )

        row_index: np.ndarray = np.arange(
            self.__height,
        )[:, None]

        column_index: np.ndarray = np.arange(
            self.__width,
        )[None, :]

        selected: np.ndarray = flat_models[
            row_index,
            column_index,
            best_index,
            :,
        ]

        self.__background = self.__hexcone_to_bgr(
            selected,
        )

        self.__finalized = True

        self.__save_background()

        return self.__background.copy()

    def __validate_configuration(self) -> None:
        """Validate SC-SOBS configuration."""

        if self.__neuron_grid_size <= 0:
            raise ValueError(
                "sc_sobs_neuron_grid_size must be > 0.",
            )

        if self.__training_frames <= 0:
            raise ValueError(
                "sc_sobs_training_frames must be > 0.",
            )

        if self.__training_threshold <= 0.0:
            raise ValueError(
                "sc_sobs_training_threshold must be > 0.",
            )

        if self.__testing_threshold <= 0.0:
            raise ValueError(
                "sc_sobs_testing_threshold must be > 0.",
            )

        if not (0.0 <= self.__testing_learning_rate <= self.__training_learning_rate <= 1.0):
            raise ValueError(
                "SC-SOBS learning rates must satisfy " "0 <= testing <= training <= 1.",
            )

        if self.__coherence_window_size <= 0 or self.__coherence_window_size % 2 == 0:
            raise ValueError(
                "sc_sobs_coherence_window_size must be " "a positive odd integer.",
            )

        if self.__update_radius < 0:
            raise ValueError(
                "sc_sobs_update_radius must be >= 0.",
            )

        if self.__gaussian_sigma <= 0.0:
            raise ValueError(
                "sc_sobs_gaussian_sigma must be > 0.",
            )

        if not self.__ground_truth_path.strip():
            raise ValueError(
                "SC-SOBS_1 requires sc_sobs_ground_truth_path. "
                "The SBI SC-SOBS_1 estimate uses the ground "
                "truth to select the final weight vector.",
            )

        if not self.__output_path.strip():
            raise ValueError(
                "sc_sobs_output_path cannot be empty.",
            )

    def __find_best_matches(
        self,
        current: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the best matching neuron for every image pixel."""

        n: int = self.__neuron_grid_size

        # -------------------------------------------------------------
        # Neural map:
        #
        # (H*n, W*n, 3)
        #
        # Reshape into:
        #
        # (H, W, n, n, 3)
        #
        # -------------------------------------------------------------
        models: np.ndarray = self.__neural_map.reshape(
            self.__height,
            n,
            self.__width,
            n,
            3,
        ).transpose(
            0,
            2,
            1,
            3,
            4,
        )

        difference: np.ndarray = (
            models
            - current[
                :,
                :,
                None,
                None,
                :,
            ]
        )

        distance_squared: np.ndarray = np.sum(
            difference * difference,
            axis=-1,
        )

        flat_distance: np.ndarray = distance_squared.reshape(
            self.__height,
            self.__width,
            n * n,
        )

        best_flat_index: np.ndarray = np.argmin(
            flat_distance,
            axis=2,
        )

        best_distance_squared: np.ndarray = np.take_along_axis(
            flat_distance,
            best_flat_index[
                :,
                :,
                None,
            ],
            axis=2,
        )[
            :,
            :,
            0,
        ]

        best_local_row: np.ndarray = (best_flat_index // n).astype(
            np.int32,
            copy=False,
        )

        best_local_column: np.ndarray = (best_flat_index % n).astype(
            np.int32,
            copy=False,
        )

        best_distance: np.ndarray = np.sqrt(
            best_distance_squared,
        )

        return (
            best_local_row,
            best_local_column,
            best_distance,
        )

    def __update_neural_map(
        self,
        current: np.ndarray,
        best_local_row: np.ndarray,
        best_local_column: np.ndarray,
        background_mask: np.ndarray,
        learning_rate: float,
    ) -> None:
        """Apply the spatially coherent self-organizing update."""

        n: int = self.__neuron_grid_size

        best_global_row: np.ndarray = self.__row_indices * n + best_local_row

        best_global_column: np.ndarray = self.__column_indices * n + best_local_column

        neural_height: int = self.__height * n

        neural_width: int = self.__width * n

        radius: int = self.__update_radius

        for delta_row in range(
            -radius,
            radius + 1,
        ):
            for delta_column in range(
                -radius,
                radius + 1,
            ):
                target_row: np.ndarray = best_global_row + delta_row

                target_column: np.ndarray = best_global_column + delta_column

                valid: np.ndarray = (
                    background_mask & (target_row >= 0) & (target_row < neural_height) & (target_column >= 0) & (target_column < neural_width)
                )

                if not np.any(valid):
                    continue

                gaussian_weight: float = self.__gaussian_weights[
                    (
                        delta_row,
                        delta_column,
                    )
                ]

                alpha: float = learning_rate * gaussian_weight

                rows: np.ndarray = target_row[valid]

                columns: np.ndarray = target_column[valid]

                observations: np.ndarray = current[valid]

                previous: np.ndarray = self.__neural_map[
                    rows,
                    columns,
                    :,
                ]

                self.__neural_map[
                    rows,
                    columns,
                    :,
                ] = (1.0 - alpha) * previous + alpha * observations

    def __build_gaussian_weights(
        self,
    ) -> dict[tuple[int, int], float]:
        """Create normalized Gaussian neural-neighborhood weights.

        The center is normalized to 1.0 so the configured c1/c2 values act
        as the maximum update coefficient.
        """

        weights: dict[
            tuple[int, int],
            float,
        ] = {}

        radius: int = self.__update_radius
        sigma_squared: float = self.__gaussian_sigma * self.__gaussian_sigma

        for delta_row in range(
            -radius,
            radius + 1,
        ):
            for delta_column in range(
                -radius,
                radius + 1,
            ):
                squared_distance: float = float(
                    delta_row * delta_row + delta_column * delta_column,
                )

                value: float = float(
                    np.exp(
                        -squared_distance / (2.0 * sigma_squared),
                    ),
                )

                weights[
                    (
                        delta_row,
                        delta_column,
                    )
                ] = value

        return weights

    def __current_threshold(self) -> float:
        """Return epsilon for the current training/testing phase."""

        if self.__frame_index < self.__training_frames:
            return self.__training_threshold

        return self.__testing_threshold

    def __current_learning_rate(self) -> float:
        """Return the current SC-SOBS learning factor.

        During training, the factor decreases from c1 toward c2.

        After the training interval, c2 is used.
        """

        if self.__frame_index >= self.__training_frames:
            return self.__testing_learning_rate

        progress: float = self.__frame_index / self.__training_frames

        learning_rate: float = self.__training_learning_rate - progress * (self.__training_learning_rate - self.__testing_learning_rate)

        return float(
            np.clip(
                learning_rate,
                0.0,
                1.0,
            ),
        )

    def __extract_central_model(
        self,
    ) -> np.ndarray:
        """Extract a non-oracle preview for display only.

        The center neuron of each n x n pixel model is displayed.

        This is not the SC-SOBS_1 SBI result.
        """

        n: int = self.__neuron_grid_size

        center: int = n // 2

        models: np.ndarray = self.__neural_map.reshape(
            self.__height,
            n,
            self.__width,
            n,
            3,
        ).transpose(
            0,
            2,
            1,
            3,
            4,
        )

        selected: np.ndarray = models[
            :,
            :,
            center,
            center,
            :,
        ]

        return self.__hexcone_to_bgr(
            selected,
        )

    def __bgr_to_hexcone(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:
        """Convert BGR uint8 image to normalized HSV-hexcone vectors."""

        frame_uint8: np.ndarray = frame.astype(
            np.uint8,
            copy=False,
        )

        hsv: np.ndarray = cv2.cvtColor(
            frame_uint8,
            cv2.COLOR_BGR2HSV,
        ).astype(
            np.float32,
        )

        # OpenCV hue:
        # H in [0, 179], corresponding approximately to [0, 358] degrees.
        angle: np.ndarray = hsv[
            :,
            :,
            0,
        ] * (2.0 * np.pi / 180.0)

        saturation: np.ndarray = (
            hsv[
                :,
                :,
                1,
            ]
            / 255.0
        )

        value: np.ndarray = (
            hsv[
                :,
                :,
                2,
            ]
            / 255.0
        )

        radial: np.ndarray = value * saturation

        x: np.ndarray = radial * np.cos(angle)

        y: np.ndarray = radial * np.sin(angle)

        return np.stack(
            (
                x,
                y,
                value,
            ),
            axis=2,
        ).astype(
            np.float32,
            copy=False,
        )

    def __hexcone_to_bgr(
        self,
        vectors: np.ndarray,
    ) -> np.ndarray:
        """Convert normalized HSV-hexcone vectors back to BGR."""

        x: np.ndarray = vectors[
            :,
            :,
            0,
        ]

        y: np.ndarray = vectors[
            :,
            :,
            1,
        ]

        value: np.ndarray = np.clip(
            vectors[
                :,
                :,
                2,
            ],
            0.0,
            1.0,
        )

        radial: np.ndarray = np.sqrt(
            x * x + y * y,
        )

        saturation: np.ndarray = np.zeros_like(
            value,
            dtype=np.float32,
        )

        nonzero_value: np.ndarray = value > 1e-8

        saturation[nonzero_value] = radial[nonzero_value] / value[nonzero_value]

        saturation = np.clip(
            saturation,
            0.0,
            1.0,
        )

        angle: np.ndarray = np.arctan2(
            y,
            x,
        )

        angle = np.mod(
            angle,
            2.0 * np.pi,
        )

        hue: np.ndarray = angle * (180.0 / (2.0 * np.pi))

        hsv: np.ndarray = np.empty(
            (
                self.__height,
                self.__width,
                3,
            ),
            dtype=np.uint8,
        )

        hsv[
            :,
            :,
            0,
        ] = np.clip(
            np.rint(hue),
            0,
            179,
        ).astype(
            np.uint8,
        )

        hsv[
            :,
            :,
            1,
        ] = np.clip(
            np.rint(
                saturation * 255.0,
            ),
            0,
            255,
        ).astype(
            np.uint8,
        )

        hsv[
            :,
            :,
            2,
        ] = np.clip(
            np.rint(
                value * 255.0,
            ),
            0,
            255,
        ).astype(
            np.uint8,
        )

        return cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR,
        )

    def __save_background(self) -> None:
        """Save the finalized SC-SOBS_1 background."""

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
                "Failed to save SC-SOBS_1 background to: " f"{absolute_output_path}",
            )

        print()
        print("=" * 60)
        print("SC-SOBS_1 BACKGROUND SAVED")
        print("=" * 60)
        print(
            f"Path       : {absolute_output_path}",
        )
        print(
            f"Frames     : {self.__frame_index}",
        )
        print(
            f"Resolution : " f"{self.__width} x {self.__height}",
        )
        print(
            "Mode       : SBI oracle extraction using ground truth",
        )
        print("=" * 60)
