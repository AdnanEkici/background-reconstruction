from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from app.data_transfer_objects.annotation_log_records import DetectedObject
from app.data_transfer_objects.annotation_log_records import FrameRecord
from app.data_transfer_objects.render_specification import RenderSpecification


class AnnotationLogger:
    """Stream frames with detections to a JSONL file.

    Each line in the file corresponds to one frame and contains its
    detected objects serialized as JSON.

    Attributes
    ----------
    __output_file_path : Path
        Path to the JSONL output file.
    __file_handle : io.TextIOWrapper
        Open file handle for writing JSONL records.
    __flush_immediately : bool
        If True, flush the file buffer after every write.
    __force_file_sync : bool
        If True, force an OS-level file sync after every write.
    """

    def __init__(
        self,
        output_file_path: str,
        flush_immediately: bool = True,
        force_file_sync: bool = False,
    ) -> None:
        """Initialize the annotation logger.

        Parameters
        ----------
        output_file_path : str, default="runs/detections.jsonl" # noqa
            Path to the JSONL output file.
        flush_immediately : bool, default=True
            Whether to flush the file buffer after every write.
        force_file_sync : bool, default=False
            Whether to call ``os.fsync`` after every flush for durability.
        """
        self.__output_file_path: Path = Path(output_file_path)
        self.__output_file_path.parent.mkdir(parents=True, exist_ok=True)

        self.__file_handle = self.__output_file_path.open("a", buffering=1, encoding="utf-8")

        self.__flush_immediately: bool = flush_immediately
        self.__force_file_sync: bool = force_file_sync

    def log_annotation(
        self,
        frame_identifier: int,
        render_specifications: list[RenderSpecification],
    ) -> None:
        """Write a JSONL record for a frame if it has detections.

        Parameters
        ----------
        frame_identifier : int
            Unique identifier of the frame (e.g., frame index).
        render_specifications : list of RenderSpecification
            Render specifications for detected objects in the frame.

        Notes
        -----
        - If no detections are present, nothing is written.
        - Each record is serialized from a :class:`FrameRecord`.
        - If ``flush_immediately`` is True, the file is flushed on write.
        - If ``force_file_sync`` is True, an OS-level sync is performed.
        """
        if not render_specifications:
            return

        frame_record: FrameRecord = FrameRecord(
            frame_id=frame_identifier,
            objects=[
                DetectedObject(bounding_box=render_specification.bounding_box, state=render_specification.state.name)
                for render_specification in render_specifications
            ],
        )

        serialized_record: str = json.dumps(asdict(frame_record), ensure_ascii=False)
        self.__file_handle.write(serialized_record + "\n")  # noqa

        if self.__flush_immediately:
            self.__file_handle.flush()
            if self.__force_file_sync:
                os.fsync(self.__file_handle.fileno())

    def close(self) -> None:
        """Close the underlying file handle safely.

        Notes
        -----
        - Silently ignores exceptions if the file is already closed.
        """
        try:
            self.__file_handle.close()
        except Exception:
            pass
