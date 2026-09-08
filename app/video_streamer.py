from __future__ import annotations

import os
from typing import Final
from typing import Iterator

import cv2
import numpy as np


class VideoStreamer:
    """Handle video streaming, display, controls, and optional output saving.

    The streamer reads frames from a video source, prepares frames for display,
    handles keyboard controls, and optionally saves the displayed output as a
    processed video.

    The displayed source frame can include the current frame ID and active
    background reconstruction method. These annotations are drawn on a copy,
    so the original frame used by the processing pipeline remains unchanged.

    Keyboard controls
    -----------------
    SPACE
        Pause or resume playback.
    K
        Save the current reconstructed background.
    Q
        Exit the application.
    ESC
        Exit the application.

    Attributes
    ----------
    SAVED_VIDEO_PATH : str
        Directory used for saved processed videos.
    BACKGROUND_IMAGE_PATH : str
        Output path used when saving the reconstructed background.
    DEFAULT_DISPLAY_SIZE : tuple[int, int]
        Default size of each displayed frame.
    RECORDING_DISPLAY_SIZE : tuple[int, int]
        Display size used when recording mode is enabled.
    """

    SAVED_VIDEO_PATH: Final[str] = "processed_videos"
    BACKGROUND_IMAGE_PATH: Final[str] = "Background.PNG"

    DEFAULT_DISPLAY_SIZE: Final[tuple[int, int]] = (896, 504)
    RECORDING_DISPLAY_SIZE: Final[tuple[int, int]] = (720, 720)

    FRAME_ID_COLOR: Final[tuple[int, int, int]] = (0, 255, 0)
    METHOD_COLOR: Final[tuple[int, int, int]] = (0, 255, 255)
    PROCESSING_TIME_COLOR: Final[tuple[int, int, int]] = (255, 255, 0)

    def __init__(
        self,
        streamer_settings: dict,
    ):
        """Initialize the video streamer.

        Parameters
        ----------
        streamer_settings : dict
            Configuration dictionary containing streamer parameters.

            Supported keys are:

            - ``"source"`` (str): Path to the input video.
            - ``"display_window_name"`` (str): OpenCV display-window name.
            - ``"enable_save"`` (bool): Whether the displayed output should
              also be written to a processed video.

        Raises
        ------
        ValueError
            If no video source is provided.

        FileNotFoundError
            If the configured video source does not exist.

        RuntimeError
            If OpenCV cannot open the configured video source.
        """
        self.__source: str | None = streamer_settings.get(
            "source",
        )

        if self.__source is None:
            raise ValueError(
                "Video source must be provided.",
            )

        if not os.path.exists(self.__source):
            raise FileNotFoundError(
                f"Video source {self.__source!r} could not be found.",
            )

        self.__video: cv2.VideoCapture = cv2.VideoCapture(
            self.__source,
        )

        if not self.__video.isOpened():
            raise RuntimeError(
                f"Could not open video source {self.__source!r}.",
            )

        self.__length: int = int(
            self.__video.get(
                cv2.CAP_PROP_FRAME_COUNT,
            ),
        )

        self.__display_size: tuple[int, int] = self.DEFAULT_DISPLAY_SIZE

        self.__display_window_name: str = streamer_settings.get(
            "display_window_name",
            "output_stream_default_window_name",
        )

        self.__save_output: bool = streamer_settings.get(
            "enable_save",
            False,
        )

        if self.__save_output:
            os.makedirs(
                self.SAVED_VIDEO_PATH,
                exist_ok=True,
            )

        self.writer: cv2.VideoWriter | None = None

        self.__sample_frame: np.ndarray | None = self.__peek()

    def __len__(self) -> int:
        """Return the total number of frames in the video source.

        Returns
        -------
        int
            Number of frames reported by the video source.
        """
        return self.__length

    @property
    def sample_frame(self) -> np.ndarray | None:
        """Return the first decoded frame without advancing the stream.

        Returns
        -------
        np.ndarray or None
            First decoded frame if available, otherwise ``None``.
        """
        return self.__sample_frame

    @property
    def source(self) -> str:
        """Return the video source filename without its extension.

        Returns
        -------
        str
            Base name of the source video.
        """
        source_name: str = os.path.splitext(
            os.path.basename(self.__source),
        )[0]

        return source_name

    @property
    def __processed_video_name(self) -> str:
        """Return the processed-video filename.

        Returns
        -------
        str
            Filename in the format ``processed_<source>.mp4``.
        """
        processed_video_name: str = f"processed_{self.source}.mp4"

        return processed_video_name

    def set_display_to_recording(self) -> None:
        """Set the display size to the recording resolution."""
        self.__display_size = self.RECORDING_DISPLAY_SIZE

    def stream(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield decoded frames sequentially from the video source.

        Yields
        ------
        tuple[int, np.ndarray]
            Frame ID and corresponding decoded frame.
        """
        frame_id: int = 0

        while True:
            ret: bool
            frame: np.ndarray | None

            ret, frame = self.read()

            if not ret or frame is None:
                break

            yield frame_id, frame

            frame_id += 1

    def display(
        self,
        *frames: np.ndarray,
        frame_id: int | None = None,
        reconstructor_method: str | None = None,
        processing_time_ms: float | None = None,
        reconstructed_background: np.ndarray | None = None,
        vertical_frames: tuple[np.ndarray, ...] = (),
    ) -> bool:
        """Display frames and process keyboard controls.

        The frame ID, active reconstruction method, and processing time are drawn
        on a copy of the first top-row frame. The original processing frame remains
        unchanged.

        Parameters
        ----------
        *frames : np.ndarray
            Frames displayed in the top row.

        frame_id : int or None, optional
            Current frame ID.

        reconstructor_method : str or None, optional
            Active background reconstruction method.

        processing_time_ms : float or None, optional
            Processing time for the current frame in milliseconds.

        reconstructed_background : np.ndarray or None, optional
            Current reconstructed background, used when saving with ``K``.

        vertical_frames : tuple[np.ndarray, ...], optional
            Frames displayed in the bottom row.

        Returns
        -------
        bool
            ``True`` to continue processing and ``False`` to terminate.
        """
        if not frames and not vertical_frames:
            raise ValueError(
                "At least one frame must be provided.",
            )

        top_frames: list[np.ndarray] = list(frames)
        bottom_frames: list[np.ndarray] = list(vertical_frames)

        if top_frames:
            annotated_frame: np.ndarray = self.__draw_frame_information(
                frame=top_frames[0],
                frame_id=frame_id,
                reconstructor_method=reconstructor_method,
                processing_time_ms=processing_time_ms,
            )

            top_frames[0] = annotated_frame

        combined_frame: np.ndarray = self.__combine_frames(
            top_frames=top_frames,
            bottom_frames=bottom_frames,
        )

        self.__show(
            frame=combined_frame,
            save_output=True,
        )

        key: int = cv2.waitKey(1) & 0xFF

        should_continue: bool = self.__handle_key(
            key=key,
            combined_frame=combined_frame,
            reconstructed_background=reconstructed_background,
            frame_id=frame_id,
        )

        return should_continue

    def read(
        self,
    ) -> tuple[bool, np.ndarray | None]:
        """Read the next video frame.

        The decoded frame is resized to ``720 x 720`` before being returned.

        Returns
        -------
        tuple[bool, np.ndarray or None]
            ``ret`` indicates whether reading succeeded. The second element is
            the decoded and resized BGR frame, or ``None`` if reading failed.
        """
        ret: bool
        frame: np.ndarray | None

        ret, frame = self.__video.read()

        if not ret or frame is None:
            return False, None

        resized_frame: np.ndarray = cv2.resize(
            frame,
            (720, 720),
        )

        return True, resized_frame

    def close(self) -> None:
        """Release video resources and destroy OpenCV windows."""
        self.__video.release()

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        cv2.destroyAllWindows()

    def __handle_key(
        self,
        key: int,
        combined_frame: np.ndarray,
        reconstructed_background: np.ndarray | None,
        frame_id: int | None,
    ) -> bool:
        """Handle keyboard controls.

        Parameters
        ----------
        key : int
            OpenCV keyboard code.

        combined_frame : np.ndarray
            Currently displayed combined frame.

        reconstructed_background : np.ndarray or None
            Current reconstructed background.

        frame_id : int or None
            Current frame identifier.

        Returns
        -------
        bool
            ``True`` to continue processing and ``False`` to terminate.
        """
        should_continue: bool = True

        if key == ord(" "):
            should_continue = self.__pause(
                combined_frame=combined_frame,
                reconstructed_background=reconstructed_background,
                frame_id=frame_id,
            )

        elif key == ord("k"):
            self.__save_background(
                reconstructed_background=reconstructed_background,
            )

        elif key == ord("q") or key == 27:
            should_continue = False

        return should_continue

    def __pause(
        self,
        combined_frame: np.ndarray,
        reconstructed_background: np.ndarray | None,
        frame_id: int | None,
    ) -> bool:
        """Pause playback until the user resumes or exits.

        While paused, the currently displayed frame remains frozen. Pressing
        ``K`` saves the reconstructed background, ``SPACE`` resumes playback,
        and ``Q`` or ``ESC`` terminates the application.

        Parameters
        ----------
        combined_frame : np.ndarray
            Frozen combined display frame.

        reconstructed_background : np.ndarray or None
            Current reconstructed background.

        frame_id : int or None
            Current frame ID.

        Returns
        -------
        bool
            ``True`` if playback should resume and ``False`` if processing
            should terminate.
        """
        if frame_id is not None:
            print(
                f"[PAUSED] Frame ID: {frame_id}",
            )
        else:
            print(
                "[PAUSED]",
            )

        should_continue: bool = True
        is_paused: bool = True

        while is_paused:
            self.__show(
                frame=combined_frame,
                save_output=False,
            )

            paused_key: int = cv2.waitKey(0) & 0xFF

            if paused_key == ord(" "):
                print(
                    "[INFO] Resuming video",
                )

                is_paused = False

            elif paused_key == ord("k"):
                self.__save_background(
                    reconstructed_background=reconstructed_background,
                )

            elif paused_key == ord("q") or paused_key == 27:
                should_continue = False
                is_paused = False

        return should_continue

    def __draw_frame_information(
        self,
        frame: np.ndarray,
        frame_id: int | None,
        reconstructor_method: str | None,
        processing_time_ms: float | None,
    ) -> np.ndarray:
        """Draw frame ID, reconstruction method, and processing time.

        The annotations are drawn on a copy so that the original input frame is
        not modified.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        frame_id : int or None
            Current frame ID.

        reconstructor_method : str or None
            Active background reconstruction method.

        processing_time_ms : float or None
            Processing time of the current frame in milliseconds.

        Returns
        -------
        np.ndarray
            Annotated copy of the input frame.
        """
        display_frame: np.ndarray = frame.copy()

        if frame_id is not None:
            cv2.putText(
                display_frame,
                f"Frame: {frame_id}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                self.FRAME_ID_COLOR,
                2,
                cv2.LINE_AA,
            )

        if reconstructor_method is not None:
            method_name: str = reconstructor_method.replace("_", " ").upper()

            cv2.putText(
                display_frame,
                f"Method: {method_name}",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.METHOD_COLOR,
                2,
                cv2.LINE_AA,
            )

        if processing_time_ms is not None:
            cv2.putText(
                display_frame,
                f"Time: {processing_time_ms:.2f} ms",
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                self.PROCESSING_TIME_COLOR,
                2,
                cv2.LINE_AA,
            )

        return display_frame

    def __combine_frames(
        self,
        top_frames: list[np.ndarray],
        bottom_frames: list[np.ndarray],
    ) -> np.ndarray:
        """Combine frames into one display image.

        Parameters
        ----------
        top_frames : list[np.ndarray]
            Frames displayed in the upper row.

        bottom_frames : list[np.ndarray]
            Frames displayed in the lower row.

        Returns
        -------
        np.ndarray
            Combined BGR display image.
        """
        top_row: np.ndarray | None = self.__horizontal_stack(
            frames=top_frames,
        )

        bottom_row: np.ndarray | None = self.__horizontal_stack(
            frames=bottom_frames,
        )

        if top_row is None and bottom_row is None:
            raise ValueError(
                "No valid images were provided for display.",
            )

        if top_row is not None and bottom_row is None:
            combined_frame: np.ndarray = top_row
            return combined_frame

        if top_row is None and bottom_row is not None:
            combined_frame = bottom_row
            return combined_frame

        top_height: int
        top_width: int
        bottom_height: int
        bottom_width: int

        top_height, top_width = top_row.shape[:2]
        bottom_height, bottom_width = bottom_row.shape[:2]

        if top_height != bottom_height:
            target_height: int = max(
                top_height,
                bottom_height,
            )

            top_row = cv2.resize(
                top_row,
                (
                    top_width,
                    target_height,
                ),
            )

            bottom_row = cv2.resize(
                bottom_row,
                (
                    bottom_width,
                    target_height,
                ),
            )

        top_width = top_row.shape[1]
        bottom_width = bottom_row.shape[1]

        if top_width < bottom_width:
            top_row = self.__pad_width(
                frame=top_row,
                target_width=bottom_width,
            )

        elif bottom_width < top_width:
            bottom_row = self.__pad_width(
                frame=bottom_row,
                target_width=top_width,
            )

        combined_frame = np.vstack(
            (
                top_row,
                bottom_row,
            ),
        )

        return combined_frame

    def __horizontal_stack(
        self,
        frames: list[np.ndarray],
    ) -> np.ndarray | None:
        """Convert, resize, and horizontally stack frames.

        Parameters
        ----------
        frames : list[np.ndarray]
            Frames to stack.

        Returns
        -------
        np.ndarray or None
            Horizontally stacked frame, or ``None`` if the input list is empty.
        """
        if not frames:
            return None

        resized_frames: list[np.ndarray] = []

        for frame in frames:
            bgr_frame: np.ndarray = self.__to_bgr(
                frame=frame,
            )

            resized_frame: np.ndarray = cv2.resize(
                bgr_frame,
                self.__display_size,
            )

            resized_frames.append(
                resized_frame,
            )

        stacked_frame: np.ndarray = np.hstack(
            resized_frames,
        )

        return stacked_frame

    @staticmethod
    def __to_bgr(
        frame: np.ndarray,
    ) -> np.ndarray:
        """Convert a grayscale frame to BGR when necessary.

        Parameters
        ----------
        frame : np.ndarray
            Input image.

        Returns
        -------
        np.ndarray
            Three-channel BGR image.
        """
        if frame.ndim == 2:
            bgr_frame: np.ndarray = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )

            return bgr_frame

        bgr_frame = frame

        return bgr_frame

    @staticmethod
    def __pad_width(
        frame: np.ndarray,
        target_width: int,
    ) -> np.ndarray:
        """Pad an image horizontally to a target width.

        Parameters
        ----------
        frame : np.ndarray
            Input frame.

        target_width : int
            Desired width.

        Returns
        -------
        np.ndarray
            Horizontally padded frame.
        """
        difference: int = target_width - frame.shape[1]

        pad_left: int = difference // 2
        pad_right: int = difference - pad_left

        padded_frame: np.ndarray = cv2.copyMakeBorder(
            frame,
            0,
            0,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        return padded_frame

    def __show(
        self,
        frame: np.ndarray,
        save_output: bool,
    ) -> None:
        """Display a frame and optionally save it to the output video.

        Parameters
        ----------
        frame : np.ndarray
            Combined display frame.

        save_output : bool
            Whether the frame should also be written to the processed video.
        """
        cv2.imshow(
            self.__display_window_name,
            frame,
        )

        if self.__save_output and save_output:
            self.__save_video_frame(
                frame=frame,
            )

    def __save_background(
        self,
        reconstructed_background: np.ndarray | None,
    ) -> None:
        """Save the current reconstructed background.

        Parameters
        ----------
        reconstructed_background : np.ndarray or None
            Background image to save.
        """
        if reconstructed_background is None:
            print(
                "[WARNING] No reconstructed background is available to save.",
            )

            return

        save_successful: bool = cv2.imwrite(
            self.BACKGROUND_IMAGE_PATH,
            reconstructed_background,
        )

        if not save_successful:
            raise RuntimeError(
                "Failed to save reconstructed background to " f"{self.BACKGROUND_IMAGE_PATH}.",
            )

        print(
            f"[INFO] Background saved to " f"{self.BACKGROUND_IMAGE_PATH}",
        )

    def __peek(
        self,
    ) -> np.ndarray | None:
        """Read the first frame without advancing the stream.

        Returns
        -------
        np.ndarray or None
            First frame if successfully decoded, otherwise ``None``.
        """
        current_position: float = self.__video.get(
            cv2.CAP_PROP_POS_FRAMES,
        )

        self.__video.set(
            cv2.CAP_PROP_POS_FRAMES,
            0,
        )

        ret: bool
        frame: np.ndarray | None

        ret, frame = self.read()

        self.__video.set(
            cv2.CAP_PROP_POS_FRAMES,
            current_position,
        )

        sample_frame: np.ndarray | None = frame if ret else None

        return sample_frame

    def __save_video_frame(
        self,
        frame: np.ndarray,
    ) -> None:
        """Write a displayed frame to the processed output video.

        Parameters
        ----------
        frame : np.ndarray
            Display frame to write.
        """
        if self.writer is None:
            height: int
            width: int

            height, width = frame.shape[:2]

            fourcc: int = cv2.VideoWriter_fourcc(
                *"mp4v",
            )

            output_path: str = os.path.join(
                self.SAVED_VIDEO_PATH,
                self.__processed_video_name,
            )

            self.writer = cv2.VideoWriter(
                output_path,
                fourcc,
                25,
                (
                    width,
                    height,
                ),
            )

        self.writer.write(
            frame,
        )
