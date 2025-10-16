"""
MRI image I/O utilities (NIfTI & MGZ formats).

Unified loader/saver built on top of nibabel, with auto-format detection
and consistent return signatures.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Union
import nibabel as nib
import numpy as np

# -------------------------------------------------------------------------
# Internal registry
# -------------------------------------------------------------------------

_SUPPORTED_EXTS = {
    ".nii": "nifti",
    ".nii.gz": "nifti",
    ".mgz": "mgz",
}

# -------------------------------------------------------------------------
# Core loading functions
# -------------------------------------------------------------------------

def _load_nib_image(
    path: Path,
    with_affine: bool = False,
    with_header: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[nib.Nifti1Header]]:
    """Internal helper for nibabel-compatible image formats."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    affine = img.affine if with_affine else None
    header = img.header if with_header else None
    return data, affine, header


def load_image(
    filepath: Union[str, Path],
    with_affine: bool = False,
    with_header: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[nib.Nifti1Header]]:
    """
    Load an MRI image (auto-detect NIfTI/MGZ).

    Parameters
    ----------
    filepath : str or Path
        Path to the image file (.nii, .nii.gz, or .mgz).
    with_affine : bool, optional
        Return affine transformation. Default False.
    with_header : bool, optional
        Return header metadata. Default False.

    Returns
    -------
    tuple
        (data, affine, header) — affine/header may be None if not requested.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If unsupported extension or invalid image.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Determine format by extension
    ext = ".nii.gz" if path.suffix == ".gz" and path.stem.endswith(".nii") else path.suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file format: {ext} (supported: {list(_SUPPORTED_EXTS)})")

    try:
        return _load_nib_image(path, with_affine, with_header)
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}") from e


# Convenience wrappers (semantic aliases)
def load_nifti(filepath, with_affine=False, with_header=False):
    """Alias for load_image() with NIfTI semantics."""
    return load_image(filepath, with_affine, with_header)


def load_mgz(filepath, with_affine=False, with_header=False):
    """Alias for load_image() with MGZ semantics."""
    return load_image(filepath, with_affine, with_header)

# -------------------------------------------------------------------------
# Saving
# -------------------------------------------------------------------------

def save_image(
    data: np.ndarray,
    filepath: Union[str, Path],
    affine: Optional[np.ndarray] = None,
    header: Optional[nib.Nifti1Header] = None,
) -> None:
    """
    Save image data to NIfTI or MGZ, auto-detected by extension.

    Parameters
    ----------
    data : np.ndarray
        Image array to save.
    filepath : str or Path
        Output path (.nii, .nii.gz, or .mgz).
    affine : np.ndarray, optional
        4x4 affine matrix. Identity if None.
    header : nib.Nifti1Header, optional
        Custom header to include.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = ".nii.gz" if path.suffix == ".gz" and path.stem.endswith(".nii") else path.suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        raise ValueError(f"Unsupported output format: {ext}")

    if affine is None:
        affine = np.eye(4)

    nib.save(nib.Nifti1Image(data, affine, header), str(path))


# Semantic wrappers
def save_nifti(data, filepath, affine=None, header=None):
    """Alias for save_image() with NIfTI semantics."""
    return save_image(data, filepath, affine, header)


def save_mgz(data, filepath, affine=None, header=None):
    """Alias for save_image() with MGZ semantics."""
    return save_image(data, filepath, affine, header)
