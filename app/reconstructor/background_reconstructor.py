# background_reconstructor.py
from __future__ import annotations

from typing import Final

import numpy as np
from reconstructor.reconstruction_methods.background_reconstructor_color_median import (
    BackgroundReconstructorColorMedian,
)
from reconstructor.reconstruction_methods.background_reconstructor_dual_ema import (
    BackgroundReconstructorDualEma,
)
from reconstructor.reconstruction_methods.background_reconstructor_labgen import (
    BackgroundReconstructorLaBGen,
)
from reconstructor.reconstruction_methods.background_reconstructor_mean import (
    BackgroundReconstructorMean,
)
from reconstructor.reconstruction_methods.background_reconstructor_mog2 import (
    BackgroundReconstructorMog2,
)
from reconstructor.reconstruction_methods.background_reconstructor_no_ema import (
    BackgroundReconstructorNoEma,
)
from reconstructor.reconstruction_methods.background_reconstructor_sc_sobs_1 import (
    BackgroundReconstructorScSobs1,
)
from reconstructor.reconstruction_methods.background_reconstructor_single_ema import (
    BackgroundReconstructorSingleEma,
)


class BackgroundReconstructor:
    """Background reconstruction method dispatcher."""

    MEAN: Final[str] = "mean"
    COLOR_MEDIAN: Final[str] = "color_median"
    MOG2: Final[str] = "mog2"
    LABGEN: Final[str] = "labgen"
    SC_SOBS_1: Final[str] = "sc_sobs_1"

    NO_EMA: Final[str] = "no_ema"
    SINGLE_EMA: Final[str] = "single_ema"
    DUAL_EMA: Final[str] = "dual_ema"

    DEFAULT_METHOD: Final[str] = DUAL_EMA

    METHODS_WITH_FOREGROUND_MASK: Final[frozenset[str]] = frozenset(
        {
            LABGEN,
            NO_EMA,
            SINGLE_EMA,
            DUAL_EMA,
        },
    )

    METHODS_WITHOUT_FOREGROUND_MASK: Final[frozenset[str]] = frozenset(
        {
            MEAN,
            COLOR_MEDIAN,
            MOG2,
            SC_SOBS_1,
        },
    )

    METHODS_REQUIRING_FINALIZATION: Final[frozenset[str]] = frozenset(
        {
            COLOR_MEDIAN,
            LABGEN,
            SC_SOBS_1,
        },
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

        method: str = str(
            background_reconstructor_settings.get(
                "method",
                self.DEFAULT_METHOD,
            ),
        ).lower()

        reconstructors = {
            self.MEAN: BackgroundReconstructorMean,
            self.COLOR_MEDIAN: (BackgroundReconstructorColorMedian),
            self.MOG2: BackgroundReconstructorMog2,
            self.LABGEN: BackgroundReconstructorLaBGen,
            self.SC_SOBS_1: BackgroundReconstructorScSobs1,
            self.NO_EMA: BackgroundReconstructorNoEma,
            self.SINGLE_EMA: (BackgroundReconstructorSingleEma),
            self.DUAL_EMA: (BackgroundReconstructorDualEma),
        }

        if method not in reconstructors:
            supported_methods: str = ", ".join(
                reconstructors.keys(),
            )

            raise ValueError(
                f"Unsupported background reconstruction " f"method: {method!r}. " f"Supported methods are: " f"{supported_methods}.",
            )

        reconstructor_class = reconstructors[method]

        self.__reconstructor = reconstructor_class(
            sample_frame=sample_frame,
            background_reconstructor_settings=(background_reconstructor_settings),
        )

        self.__method: str = method

    def __call__(
        self,
        frame: np.ndarray,
        binary_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Process one frame."""

        if self.requires_foreground_mask and binary_mask is None:
            raise ValueError(
                f"Reconstruction method " f"{self.__method!r} requires " f"a foreground binary mask.",
            )

        return self.__reconstructor(
            frame=frame,
            binary_mask=binary_mask,
        )

    @property
    def method(self) -> str:
        return self.__method

    @property
    def requires_foreground_mask(self) -> bool:
        return self.__method in self.METHODS_WITH_FOREGROUND_MASK

    @property
    def requires_finalization(self) -> bool:
        return self.__method in self.METHODS_REQUIRING_FINALIZATION

    def finalize(self) -> np.ndarray:
        """Finalize a batch/oracle reconstruction method."""

        finalize_method = getattr(
            self.__reconstructor,
            "finalize",
            None,
        )

        if finalize_method is None:
            raise RuntimeError(
                f"Reconstruction method " f"{self.__method!r} does not " f"require finalization.",
            )

        return finalize_method()
