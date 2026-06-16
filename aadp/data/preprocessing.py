from typing import Tuple

import numpy as np
import SimpleITK as sitk
import torch


def window_and_normalize(
    volume: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> torch.Tensor:
    """Clip to HU window and linearly normalise to [0, 1].

    Args:
        volume:  (D, H, W) float32 numpy array in raw HU values.
        hu_min:  Lower HU boundary (default -1000, start of lung window).
        hu_max:  Upper HU boundary (default  400, end of lung window).

    Returns:
        (D, H, W) float32 torch.Tensor with values in [0, 1].
    """
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / (hu_max - hu_min)
    return torch.from_numpy(volume.astype(np.float32))


def resample_to_isotropic(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    target_spacing: float = 1.0,
    interpolator=sitk.sitkLinear,
    pad_value: float = -1000.0,
) -> np.ndarray:
    """Resample a CT volume to isotropic voxel spacing using SimpleITK.

    CT volumes are anisotropic by design: in-plane spacing (0.5–0.75 mm) is
    much finer than slice thickness (3–5 mm). This resamples all axes to a
    common target_spacing so downstream models see uniform geometry.

    Args:
        volume:         (D, H, W) float32 numpy array (axes: Z, Y, X).
        spacing:        Current voxel spacing as (z_mm, y_mm, x_mm).
        target_spacing: Desired isotropic spacing in mm (default 1.0 mm).
        interpolator:   SimpleITK interpolator (default linear; use
                        sitk.sitkNearestNeighbor for label maps).
        pad_value:      Fill value for voxels outside the original FOV
                        (default -1000 = HU air).

    Returns:
        (D', H', W') float32 numpy array resampled to target_spacing.
    """
    # SimpleITK stores images as (X, Y, Z) internally; GetImageFromArray
    # takes a (Z, Y, X) array and maps it correctly.
    sitk_img = sitk.GetImageFromArray(volume)
    # SetSpacing expects (x, y, z) order.
    sitk_img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))

    orig_size = sitk_img.GetSize()        # (X, Y, Z)
    orig_spacing = sitk_img.GetSpacing()  # (x, y, z)

    new_size = [
        int(round(sz * sp / target_spacing))
        for sz, sp in zip(orig_size, orig_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([target_spacing] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(pad_value)
    resampler.SetInterpolator(interpolator)

    resampled = resampler.Execute(sitk_img)
    # GetArrayFromImage returns (Z, Y, X) = (D, H, W).
    return sitk.GetArrayFromImage(resampled).astype(np.float32)


def pad_or_crop_depth(
    volume: np.ndarray,
    target_D: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Centre-crop or zero-pad a volume along the depth axis to target_D slices.

    Cropping and padding are both centred so anatomy stays roughly aligned
    across volumes of different depths.

    Args:
        volume:   (D, H, W) numpy array.
        target_D: Desired number of depth slices.
        pad_value: Fill value used when padding (default 0.0, i.e. the
                   normalised equivalent of the lower HU boundary after
                   window_and_normalize has been applied).

    Returns:
        (target_D, H, W) numpy array.
    """
    D = volume.shape[0]
    if D == target_D:
        return volume

    if D > target_D:
        # Centre-crop: discard equal numbers of slices from each end.
        start = (D - target_D) // 2
        return volume[start : start + target_D]

    # Centre-pad: add equal numbers of fill slices to each end.
    pad_before = (target_D - D) // 2
    pad_after = target_D - D - pad_before
    pad_width = [(pad_before, pad_after)] + [(0, 0)] * (volume.ndim - 1)
    return np.pad(volume, pad_width, mode="constant", constant_values=pad_value)
