from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def calculate_metrics(
    ground_truth_path: str,
    reconstructed_path: str,
) -> None:
    ground_truth = cv2.imread(ground_truth_path)
    reconstructed = cv2.imread(reconstructed_path)

    if ground_truth is None:
        raise FileNotFoundError(
            f"Could not read ground truth: {ground_truth_path}",
        )

    if reconstructed is None:
        raise FileNotFoundError(
            f"Could not read reconstructed image: {reconstructed_path}",
        )

    # Resize reconstructed image to match ground-truth resolution.
    gt_height, gt_width = ground_truth.shape[:2]

    reconstructed = cv2.resize(
        reconstructed,
        (gt_width, gt_height),
        interpolation=cv2.INTER_LINEAR,
    )

    gt = ground_truth.astype(np.float64)
    rec = reconstructed.astype(np.float64)

    mse = np.mean((gt - rec) ** 2)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10((255.0**2) / mse)

    ssim = structural_similarity(
        ground_truth,
        reconstructed,
        channel_axis=2,
        data_range=255,
    )

    print(f"MSE       : {mse:.6f}")
    print(f"PSNR (dB) : {psnr:.6f}")
    print(f"SSIM      : {ssim:.6f}")


if __name__ == "__main__":
    calculate_metrics(
        ground_truth_path="test_videos/groundtruths/GT_HighwayII.png",
        reconstructed_path="Background.PNG",
    )
