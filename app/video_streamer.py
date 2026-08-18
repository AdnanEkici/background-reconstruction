from __future__ import annotations

import os
from typing import Final
from typing import Iterator

import cv2
import numpy as np


class VideoStreamer:
    """Handle video streaming, frame extraction, display, and optional saving.

    This class provides utilities to read frames from a video source, resize and display
    them in real-time, and optionally save output frames as a video. It supports
    configurable display scaling and window naming via settings.

    Attributes
    ----------
    SAVED_VIDEO_PATH : str
        Directory where processed videos are saved.
    __source : str or None
        Path to the input video file.
    __video : cv2.VideoCapture
        OpenCV video capture object for the source.
    __length : int
        Total number of frames in the source.
    __sample_frame : np.ndarray or None
        First frame of the video, used as a reference.
    __display_size : tuple[int, int]
        Fixed display size for all frames shown in the display window.
    __display_window_name : str
        Window name used for display.
    __save_output : bool
        Whether to save displayed frames to disk.
    writer : cv2.VideoWriter or None
        Video writer object for saving output video.
    """

    SAVED_VIDEO_PATH: Final[str] = "processed_videos"

    def __init__(self, streamer_settings: dict):
        """Initialize the video streamer.

        Parameters
        ----------
        streamer_settings : dict
            Configuration dictionary with keys:

            - ``source`` : str
                Path to the input video file.
            - ``display_scale_percent`` : int
                Percentage to scale displayed frames (currently unused, fixed size).
            - ``display_window_name`` : str
                Name of the OpenCV display window.
            - ``enable_save`` : bool
                If True, saves displayed frames to disk.
        """

        self.__source: str | None = streamer_settings.get("source")
        if not os.path.exists(self.__source):
            raise FileNotFoundError(f"Video source {self.__source} could not found !")

        self.__video = cv2.VideoCapture(self.__source)
        self.__length: int = int(self.__video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__sample_frame: np.ndarray | None = self.__peek()
        self.__display_size = (896, 504)
        self.__display_window_name: str = streamer_settings.get(
            "display_window_name", "output_stream_default_window_name"
        )
        self.__save_output: bool = streamer_settings.get("enable_save", False)
        self.__save_output and os.makedirs(self.SAVED_VIDEO_PATH, exist_ok=True)
        self.writer = None

    def set_display_to_recording(self):
        self.__display_size = (720, 720)

    @property
    def sample_frame(self):
        """Return the first frame of the video.

        Returns
        -------
        np.ndarray or None
            First frame of the video if available, else ``None``.
        """
        return self.__sample_frame

    def stream(self) -> Iterator[tuple[int, np.ndarray]]:
        """Generator that yields frames sequentially from the video source.

        Yields
        ------
        tuple[int, np.ndarray]
            Frame ID and the corresponding frame image.
        """
        frame_id: int = 0
        while True:
            ret: bool
            frame: np.ndarray
            ret, frame = self.read()
            if not ret:
                break
            yield frame_id, frame
            frame_id += 1

    def display(self, *frames: np.ndarray, vertical_frames: tuple[np.ndarray, ...] = ()) -> None:
        """Display frames in a window, arranged in two rows.

        - Top row: frames passed via ``*frames``
        - Bottom row: frames passed via ``vertical_frames``

        If only one row is provided, only that row is shown. Frames are resized to a
        fixed size and stacked accordingly. Optionally, the combined frame is also saved.

        Parameters
        ----------
        *frames : np.ndarray
            Frames for the top row.
        vertical_frames : tuple of np.ndarray, optional
            Frames for the bottom row.

        Raises
        ------
        ValueError
            If no frames are provided for either row.
        """
        if not frames and not vertical_frames:
            raise ValueError("At least one image must be provided (either frames or vertical_frames).")

        def to_bgr(img: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

        def resize_all(imgs: list[np.ndarray]) -> list[np.ndarray]:
            return [cv2.resize(to_bgr(im), self.__display_size) for im in imgs]

        def hstack_or_none(imgs: list[np.ndarray]) -> np.ndarray | None:
            if not imgs:
                return None
            return np.hstack(resize_all(imgs))

        top_row = hstack_or_none(list(frames))
        bottom_row = hstack_or_none(list(vertical_frames))

        if top_row is None and bottom_row is None:
            raise ValueError("No valid images after processing.")

        if top_row is not None and bottom_row is None:
            combined = top_row
        elif top_row is None and bottom_row is not None:
            combined = bottom_row
        else:
            h_top, w_top = top_row.shape[:2]
            h_bot, w_bot = bottom_row.shape[:2]

            if h_top != h_bot:
                target_h = max(h_top, h_bot)
                top_row = cv2.resize(top_row, (w_top, target_h))
                bottom_row = cv2.resize(bottom_row, (w_bot, target_h))
                h_top, w_top = top_row.shape[:2]
                h_bot, w_bot = bottom_row.shape[:2]

            if w_top != w_bot:
                pad_left, pad_right = 0, 0
                if w_top < w_bot:
                    diff = w_bot - w_top
                    pad_left, pad_right = diff // 2, diff - diff // 2
                    top_row = cv2.copyMakeBorder(
                        top_row, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                else:
                    diff = w_top - w_bot
                    pad_left, pad_right = diff // 2, diff - diff // 2
                    bottom_row = cv2.copyMakeBorder(
                        bottom_row, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )

            combined = np.vstack([top_row, bottom_row])

        cv2.imshow(self.__display_window_name, combined)

        if self.__save_output:
            self.__save(frame=combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            print("Exiting...")
            exit(0)

    def __len__(self) -> int:
        """Return the total number of frames in the video source.

        Returns
        -------
        int
            Total frame count.
        """
        return self.__length

    def __peek(self) -> np.ndarray | None:
        """Read the first frame of the video without advancing the capture position.

        Returns
        -------
        np.ndarray or None
            First frame of the video if available, else ``None``.
        """
        pos = self.__video.get(cv2.CAP_PROP_POS_FRAMES)
        self.__video.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = self.read()
        self.__video.set(cv2.CAP_PROP_POS_FRAMES, pos)

        return frame if ret else None

    def __save(self, frame: np.ndarray) -> None:
        """Save a frame to the output video file.

        Parameters
        ----------
        frame : np.ndarray
            Frame to save (BGR image).
        """
        if self.writer is None:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.SAVED_VIDEO_PATH + os.sep + self.__processed_video_name, fourcc, 25, (width, height)
            )
        self.writer.write(frame)

    @property
    def __processed_video_name(self) -> str:
        """Return the filename for the processed video.

        Returns
        -------
        str
            Output video filename in the format ``processed_<source>.mp4``.
        """
        source_name = os.path.basename(self.__source).split(".")[0]
        return "processed_" + source_name + ".mp4"

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame from the video source.

        The frame is automatically resized to a fixed resolution of 720x720
        if successfully read.

        Returns
        -------
        tuple[bool, np.ndarray or None]
            - ``ret`` : bool
                True if a frame was successfully read, False if end of stream.
            - ``frame`` : np.ndarray or None
                The decoded frame as a 720x720 BGR image if successful,
                otherwise None.
        """
        ret, frame = self.__video.read()
        if not ret or frame is None:
            return False, None
        frame = cv2.resize(frame, (720, 720))
        return True, frame

    @property
    def source(self) -> str:
        """
        Name of the video source file without extension.

        Returns
        -------
        str
            Base name of the video file (without directory path and extension).
            For example, if the source path is ``/videos/input/sample.mp4``,
            this property returns ``"sample"``.
        """
        return os.path.basename(self.__source).split(".")[0]
