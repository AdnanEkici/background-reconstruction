from __future__ import annotations

from typing import Final

import cv2
import numpy as np
import torch

import app.utilities.utils as utils

# NOTE: This code is from https://github.com/vandroogenbroeckmarc/vibe/tree/main/Python
# here slight changes has been done.


class ViBe:
    """Sample-based background subtraction algorithm for motion detection in video.

    Implements the ViBe algorithm with support for GPU/CPU execution,
    batch inference, and post-processing using median filtering and
    morphological operations.

    Attributes
    ----------
    CPU : str
        Default device identifier for CPU (``"cpu"``).
    __device : str
        Device used for computation (``"cpu"`` or ``"cuda"``).
    __number_of_samples : int
        Number of background samples stored per pixel.
    __matching_threshold : int
        Pixel intensity threshold for considering a match.
    __matching_number : int
        Number of matches required to classify a pixel as background.
    __update_factor : int
        Probability factor controlling background model updates.
    __neighborhood_radius : int
        Radius for neighboring pixel sampling during updates.
    __median_filter_kernel_size : int
        Kernel size for median filtering the binary mask.
    __morphological_opening_structuring_element_size : tuple[int, int]
        Size of the structuring element for morphological operations.
    __history_buffer : torch.Tensor
        Storage buffer holding background samples for each pixel.
    """

    CPU: Final[str] = "cpu"

    def __init__(self, calibration_frame: np.ndarray, background_subtractor_settings: dict):
        """Initialize the ViBe background subtractor.

        Parameters
        ----------
        calibration_frame : np.ndarray
            Initial RGB frame used to initialize the background model.
            Shape ``(H, W, 3)``.
        background_subtractor_settings : dict
            Configuration parameters:

            - ``device`` (str): Device identifier (``"cpu"`` or ``"cuda"``). Default is ``"cpu"``.
            - ``number_of_samples`` (int): Number of background samples stored per pixel. Default 30.
            - ``matching_threshold`` (int): Pixel intensity threshold for matching. Default 10.
            - ``matching_number`` (int): Number of matches required to classify as background. Default 2.
            - ``update_factor`` (int): Probability factor for background updates. Default 2.
            - ``neighborhood_radius`` (int): Radius for neighboring pixel sampling. Default 1.
            - ``median_filter_kernel_size`` (int): Kernel size for median filtering. Default 3.
            - ``morphological_opening_structuring_element_size`` (int): Structuring element size for
              morphological opening/closing. Default 7.
        """
        self.__device: str = background_subtractor_settings.get("device", self.CPU)
        self.__device = utils.safe_set_device(device=self.__device)

        self.__number_of_samples: int = background_subtractor_settings.get("number_of_samples", 30)
        self.__matching_threshold: int = background_subtractor_settings.get("matching_threshold", 10)
        self.__matching_number: int = background_subtractor_settings.get("matching_number", 2)
        self.__update_factor: int = background_subtractor_settings.get("update_factor", 2)
        self.__neighborhood_radius: int = background_subtractor_settings.get("neighborhood_radius", 1)
        self.__median_filter_kernel_size: int = background_subtractor_settings.get("median_filter_kernel_size", 3)
        self.__morphological_opening_structuring_element_size: tuple[int, int] = (
            background_subtractor_settings.get("morphological_opening_structuring_element_size", 7),
        ) * 2  # Create kernel size tuple (n, n)

        self.__initialize(calibration_frame=calibration_frame)

    def __call__(self, frame: np.ndarray):
        """Perform foreground segmentation on an input frame.

        Parameters
        ----------
        frame : np.ndarray
            Input RGB frame. Shape ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            Binary mask where foreground pixels are 255 and background pixels are 0.
            Shape ``(H, W)``, dtype=uint8.
        """
        frame_tensor: torch.Tensor = self.__numpy_to_torch_tensor(frame=frame)
        mask: torch.Tensor = self.__segmentation(frame_tensor)
        self.__update(frame_tensor, mask)
        mask: np.ndarray = self.__torch_tensor_to_numpy(frame=mask)
        mask = cv2.medianBlur(mask, ksize=self.__median_filter_kernel_size)
        mask = self.__apply_morphological_close(mask=mask)
        return mask

    def __initialize(self, calibration_frame):
        """Initialize internal buffers and background model using a calibration frame.

        Parameters
        ----------
        calibration_frame : np.ndarray
            RGB frame used to initialize the ViBe model. Shape ``(H, W, 3)``.
        """
        frame_tensor = self.__numpy_to_torch_tensor(calibration_frame)

        self.__channels: int = frame_tensor.size()[0]
        self.__height: int = frame_tensor.size()[1]
        self.__width: int = frame_tensor.size()[2]

        # Storage for the history
        self.__history_buffer: torch.Tensor = torch.zeros(
            frame_tensor.size()[0],
            frame_tensor.size()[1],
            frame_tensor.size()[2],
            self.__number_of_samples,
            dtype=torch.float,
            device=self.__device,
        )

        # Buffers with random values
        self.__update_mask: torch.Tensor = (
            torch.empty(self.__width * self.__height, dtype=torch.float).uniform_(0, 1).to(self.__device)
        )
        self.__neighbor_row: torch.Tensor | None = None
        self.__neighbor_col: torch.Tensor | None = None
        self.__position: torch.Tensor | None = None

        # Some other precomputations
        self.__row: torch.Tensor = (
            torch.arange(0, self.__height, dtype=torch.float, device=self.__device)
            .repeat(self.__width, 1)
            .transpose(0, 1)
        )
        self.__col: torch.Tensor = torch.arange(0, self.__width, dtype=torch.float, device=self.__device).repeat(
            self.__height, 1
        )

        # Threshold values
        self.__one: torch.Tensor = torch.zeros(self.__update_mask.size(), dtype=torch.float, device=self.__device) + 1
        self.__zero: torch.Tensor = torch.zeros(self.__update_mask.size(), dtype=torch.float, device=self.__device)
        self.__BG: torch.Tensor = torch.zeros(frame_tensor.size(), dtype=torch.float, device=self.__device)
        self.__FG: torch.Tensor = torch.zeros(frame_tensor.size(), dtype=torch.float, device=self.__device) + 255
        self.__BG1: torch.Tensor = torch.zeros(frame_tensor.size()[1:3], dtype=torch.float, device=self.__device)
        self.__FG1: torch.Tensor = torch.zeros(frame_tensor.size()[1:3], dtype=torch.float, device=self.__device) + 1

        for test in np.arange(self.__matching_number):
            self.__history_buffer[:, :, :, test] = frame_tensor

        for test in np.arange(self.__number_of_samples - self.__matching_number) + self.__matching_number:
            noise: torch.Tensor = torch.randint(-20, 20, frame_tensor.size()).to(self.__device).type(torch.float)
            value_plus_noise: torch.Tensor = frame_tensor + noise
            value_plus_noise = torch.where(value_plus_noise > 255, self.__FG, value_plus_noise)
            value_plus_noise = torch.where(value_plus_noise < 0, self.__BG, value_plus_noise)
            self.__history_buffer[:, :, :, test] = value_plus_noise

        self.__update_mask = torch.where(self.__update_mask > 1 / self.__update_factor, self.__zero, self.__one)

        amount: int = int(torch.sum(self.__update_mask).to(self.CPU).numpy())

        self.__neighbor_row: torch.Tensor = (
            torch.randint(-self.__neighborhood_radius, self.__neighborhood_radius + 1, (amount,))
            .to(self.__device)
            .type(torch.float)
        )
        self.__neighbor_col: torch.Tensor = (
            torch.randint(-self.__neighborhood_radius, self.__neighborhood_radius + 1, (amount,))
            .to(self.__device)
            .type(torch.float)
        )

        self.__position: torch.Tensor = (
            torch.randint(0, self.__number_of_samples, (amount,)).to(self.__device).type(torch.float)
        )

    def __apply_morphological_close(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological opening and closing to clean the binary mask.

        Parameters
        ----------
        mask : np.ndarray
            Input binary mask. Shape ``(H, W)``, dtype=uint8.

        Returns
        -------
        np.ndarray
            Cleaned binary mask after opening and closing.
        """
        kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.__morphological_opening_structuring_element_size
        )
        mask_cleaned: np.ndarray = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
        _, binary_mask = cv2.threshold(mask_cleaned, 127, 255, cv2.THRESH_BINARY)
        return binary_mask

    def __segmentation(self, frame: torch.Tensor) -> torch.Tensor:
        """Perform background subtraction using the history buffer.

        Parameters
        ----------
        frame : torch.Tensor
            Current frame tensor. Shape ``(C, H, W)``, dtype=float32.

        Returns
        -------
        torch.Tensor
            Binary segmentation map as a tensor with 1 for foreground and 0 for background.
            Shape ``(H, W)``, dtype=float32.
        """

        matching_threshold: float = 4.5 * self.__matching_threshold
        delta: torch.Tensor = frame.unsqueeze(-1).repeat(1, 1, 1, self.__number_of_samples) - self.__history_buffer
        num_matches: torch.Tensor = torch.sum(
            torch.where(
                torch.sum(torch.abs(delta), 0) <= matching_threshold,
                self.__FG1.unsqueeze(-1).repeat(1, 1, self.__number_of_samples),
                self.__BG1.unsqueeze(-1).repeat(1, 1, self.__number_of_samples),
            ),
            dim=-1,
        )
        segmentation_map: torch.Tensor = torch.where(num_matches >= self.__matching_number, self.__BG1, self.__FG1)

        return segmentation_map

    def __update(self, frame, updating_mask):
        """Update the background model based on static pixels.

        Parameters
        ----------
        frame : torch.Tensor
            Current frame tensor. Shape ``(C, H, W)``, dtype=float32.
        updating_mask : torch.Tensor
            Binary mask indicating which pixels are foreground. Shape ``(H, W)``, dtype=float32.
        """
        r: int = int(torch.randint(0, self.__update_mask.size()[0], (1,)).numpy())
        r2 = torch.randint(0, self.__position.size()[0], (3,)).numpy().astype(int)

        update_frame: torch.Tensor = self.__roll(self.__update_mask, r).view(self.__height, self.__width)
        self.__neighbor_row = self.__roll(self.__neighbor_row, r2[0])
        self.__neighbor_col = self.__roll(self.__neighbor_col, r2[1])
        self.__position = self.__roll(self.__position, r2[2])

        update: torch.Tensor = update_frame * (1 - updating_mask)
        num_updates: int = int(torch.sum(update).to(self.CPU).numpy())

        if self.__channels == 3:
            row: torch.Tensor = self.__row[update == 1]
            col: torch.Tensor = self.__col[update == 1]
            pos: torch.Tensor = self.__position[0:num_updates]

            self.__history_buffer[
                :, row.type(torch.LongTensor), col.type(torch.LongTensor), pos.type(torch.LongTensor)
            ] = frame[:, row.type(torch.LongTensor), col.type(torch.LongTensor)]

            row_shift: torch.Tensor = row + self.__neighbor_row[0:num_updates]
            row_shift = torch.where(
                row_shift >= self.__height, self.__one[0:num_updates] * self.__height - 1, row_shift
            )
            row_shift = torch.where(row_shift < 0, self.__zero[0:num_updates], row_shift)

            col_shift: torch.Tensor = col + self.__neighbor_col[0:num_updates]
            col_shift = torch.where(col_shift >= self.__width, self.__one[0:num_updates] * self.__width - 1, col_shift)
            col_shift = torch.where(col_shift < 0, self.__zero[0:num_updates], col_shift)

            pos = self.__roll(self.__position, r2[0])[0:num_updates]

            self.__history_buffer[
                :, row_shift.type(torch.LongTensor), col_shift.type(torch.LongTensor), pos.type(torch.LongTensor)
            ] = frame[:, row.type(torch.LongTensor), col.type(torch.LongTensor)]

    def __roll(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Roll (circularly shift) a tensor along its first dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        n : int
            Number of positions to roll.

        Returns
        -------
        torch.Tensor
            Rolled tensor.
        """
        return torch.cat((x[-n:], x[:-n]))

    def __numpy_to_torch_tensor(self, frame: np.ndarray | None) -> torch.Tensor:
        """Convert a NumPy RGB image to a torch tensor.

        Parameters
        ----------
        frame : np.ndarray or None
            Input RGB frame. Shape ``(H, W, 3)``, dtype=uint8.
            If ``None``, returns ``None``.

        Returns
        -------
        torch.Tensor or None
            Tensor of shape ``(C, H, W)``, dtype=float32 on the configured device,
            or ``None`` if input is None.
        """
        return torch.from_numpy(frame).permute(2, 0, 1).float().to(self.__device) if frame is not None else None

    def __torch_tensor_to_numpy(self, frame: torch.Tensor | None) -> np.ndarray:
        """Convert a torch tensor mask back to a NumPy binary mask.

        Parameters
        ----------
        frame : torch.Tensor or None
            Input torch tensor. Shape ``(H, W)``, dtype=float32 or float.

        Returns
        -------
        np.ndarray or None
            Binary mask as NumPy array (0 or 255 values), dtype=uint8.
            Returns ``None`` if input is None.
        """
        return frame.type(torch.uint8).cpu().numpy() * 255 if frame is not None else None
