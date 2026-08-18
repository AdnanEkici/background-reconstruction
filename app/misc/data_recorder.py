from __future__ import annotations

import argparse
import os
import cv2
import numpy as np
import app.utilities.utils as utils
from app.reconstructor.background_reconstructor import BackgroundReconstructor
from app.subtractor.vibe_background_subtraction import ViBe
from app.video_streamer import VideoStreamer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run application with configuration file")
    parser.add_argument(
        "--configuration_file_path",
        type=str,
        default="configuration_files" + os.sep + "configuration_1.yml",
        help="Path to the configuration YAML file",
    )
    return parser.parse_args()




if __name__ == "__main__":
    args = parse_args()
    configuration_file_path: str = args.configuration_file_path
    application_settings: dict = utils.read_yaml(yaml_file_path=configuration_file_path)

    streamer_settings = application_settings.get("streamer_settings")
    background_subtractor_settings = application_settings.get("background_subtractor_settings")
    background_reconstructor_settings = application_settings.get("background_reconstructor_settings")

    streamer = VideoStreamer(streamer_settings=streamer_settings)
    streamer.set_display_to_recording()
    calibration_frame = streamer.sample_frame

    vibe = ViBe(calibration_frame=calibration_frame, background_subtractor_settings=background_subtractor_settings)
    background_reconstructor = BackgroundReconstructor(
        sample_frame=calibration_frame, background_reconstructor_settings=background_reconstructor_settings
    )


    for frame_id, frame in streamer.stream():
        binary_mask = vibe(frame)
        static_foreground_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        reconstructed_background = background_reconstructor(frame, binary_mask)
        streamer.display(reconstructed_background)

        key = cv2.waitKey(1)
        if key == ord(" "):
            print(f"[INFO] Frame ID: {frame_id}")

    cv2.destroyAllWindows()
