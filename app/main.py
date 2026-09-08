from __future__ import annotations

import argparse
import os
import random
from time import perf_counter

import cv2
import numpy as np
import torch

import app.utilities.utils as utils
from app.reconstructor.background_reconstructor import BackgroundReconstructor
from app.subtractor.vibe_background_subtraction import ViBe
from app.video_streamer import VideoStreamer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run application with configuration file",
    )

    parser.add_argument(
        "--configuration_file_path",
        type=str,
        default="configuration_files" + os.sep + "configuration_1.yml",
        help="Path to the configuration YAML file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible experiments",
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cv2.setRNGSeed(seed)


if __name__ == "__main__":
    args = parse_args()

    configuration_file_path: str = args.configuration_file_path

    set_random_seed(
        seed=args.seed,
    )

    application_settings: dict = utils.read_yaml(
        yaml_file_path=configuration_file_path,
    )

    streamer_settings = application_settings.get(
        "streamer_settings",
    )

    background_subtractor_settings = application_settings.get(
        "background_subtractor_settings",
    )

    background_reconstructor_settings = application_settings.get(
        "background_reconstructor_settings",
    )

    streamer = VideoStreamer(
        streamer_settings=streamer_settings,
    )

    calibration_frame = streamer.sample_frame

    vibe = ViBe(
        calibration_frame=calibration_frame,
        background_subtractor_settings=background_subtractor_settings,
    )

    background_reconstructor = BackgroundReconstructor(
        sample_frame=calibration_frame,
        background_reconstructor_settings=background_reconstructor_settings,
    )

    total_processing_time_ms: float = 0.0
    processed_frame_count: int = 0

    try:
        for frame_id, frame in streamer.stream():
            start = perf_counter()

            binary_mask = vibe(frame)

            reconstructed_background = background_reconstructor(
                frame,
                binary_mask,
            )

            end = perf_counter()

            processing_time_ms: float = (end - start) * 1000.0

            total_processing_time_ms += processing_time_ms
            processed_frame_count += 1

            should_continue: bool = streamer.display(
                frame,
                reconstructed_background,
                frame_id=frame_id,
                reconstructor_method=background_reconstructor.method,
                processing_time_ms=processing_time_ms,
                reconstructed_background=reconstructed_background,
            )

            if not should_continue:
                break

        if background_reconstructor.requires_finalization:
            reconstructed_background = background_reconstructor.finalize()

    finally:
        streamer.close()

        if processed_frame_count > 0:
            average_processing_time_ms: float = (
                total_processing_time_ms / processed_frame_count
            )

            average_fps: float = 1000.0 / average_processing_time_ms

            print()
            print("=" * 50)
            print("RUNTIME SUMMARY")
            print("=" * 50)
            print(
                f"Method       : {background_reconstructor.method}",
            )
            print(
                f"Seed         : {args.seed}",
            )
            print(
                f"Frames       : {processed_frame_count}",
            )
            print(
                f"Average time : {average_processing_time_ms:.2f} ms/frame",
            )
            print(
                f"Average FPS  : {average_fps:.2f}",
            )
            print("=" * 50)