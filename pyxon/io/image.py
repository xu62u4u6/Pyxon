"""
Object-oriented MRI image interface.

This module provides a high-level object-oriented API for working with MRI images,
built on top of the low-level image_loader functions.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np

from .image_loader import load_image, save_nifti, save_mgz


class MRIImage:
    """
    High-level interface for MRI images.

    This class wraps nibabel images with convenient methods for common operations.
    It maintains the data, affine transformation, and header information together.

    Parameters
    ----------
    data : np.ndarray
        The image data array.
    affine : np.ndarray
        The 4x4 affine transformation matrix.
    header : nibabel.Nifti1Header, optional
        The image header. If None, a default header is created.
    filepath : str or Path, optional
        Original filepath if loaded from disk.

    Attributes
    ----------
    data : np.ndarray
        The image data array.
    affine : np.ndarray
        The 4x4 affine transformation matrix.
    header : nibabel.Nifti1Header
        The image header.
    filepath : Path or None
        Original filepath if loaded from disk.
    shape : tuple
        Shape of the image data.
    ndim : int
        Number of dimensions.

    Examples
    --------
    >>> # Load an image
    >>> img = MRIImage.load('brain.nii.gz')
    >>> print(img.shape)
    (256, 256, 256)
    
    >>> # Access data and affine
    >>> data = img.data
    >>> affine = img.affine
    
    >>> # Get a slice
    >>> slice_data = img.get_slice(axis=2, index=128)
    
    >>> # Save to new file
    >>> img.save('output.nii.gz')
    """

    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        header: Optional[nib.Nifti1Header] = None,
        filepath: Optional[Union[str, Path]] = None,
    ):
        self.data = data
        self.affine = affine
        self.header = header
        self.filepath = Path(filepath) if filepath else None

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MRIImage":
        """
        Load an MRI image from file.

        Parameters
        ----------
        filepath : str or Path
            Path to the image file (.nii, .nii.gz, or .mgz).

        Returns
        -------
        MRIImage
            Loaded image object.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> img = MRIImage.load('brain.mgz')
        """
        data, affine, header = load_image(
            filepath, with_affine=True, with_header=True
        )
        return cls(data, affine, header, filepath)

    @classmethod
    def from_nibabel(cls, nib_img: nib.Nifti1Image) -> "MRIImage":
        """
        Create MRIImage from a nibabel image object.

        Parameters
        ----------
        nib_img : nibabel.Nifti1Image
            A nibabel image object.

        Returns
        -------
        MRIImage
            Image object wrapping the nibabel image.

        Examples
        --------
        >>> nib_img = nib.load('brain.nii.gz')
        >>> img = MRIImage.from_nibabel(nib_img)
        """
        data = np.asarray(nib_img.dataobj)
        return cls(data, nib_img.affine, nib_img.header)

    def to_nibabel(self) -> nib.Nifti1Image:
        """
        Convert to a nibabel image object.

        Returns
        -------
        nibabel.Nifti1Image
            Nibabel image representation.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> nib_img = img.to_nibabel()
        >>> nib.save(nib_img, 'output.nii.gz')
        """
        return nib.Nifti1Image(self.data, self.affine, self.header)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the image to a file.

        The format is determined by the file extension (.nii, .nii.gz, or .mgz).

        Parameters
        ----------
        filepath : str or Path
            Path where the image will be saved.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> img.save('output.nii.gz')
        >>> img.save('output.mgz')
        """
        filepath = Path(filepath)
        if filepath.suffix == '.mgz':
            save_mgz(self.data, filepath, self.affine, self.header)
        else:
            save_nifti(self.data, filepath, self.affine, self.header)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the image data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Get the number of dimensions."""
        return self.data.ndim

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        """
        Extract a 2D slice from the 3D image.

        Parameters
        ----------
        axis : int
            Axis along which to slice (0, 1, or 2).
        index : int
            Index of the slice to extract.

        Returns
        -------
        np.ndarray
            2D array representing the slice.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> axial_slice = img.get_slice(axis=2, index=128)
        >>> sagittal_slice = img.get_slice(axis=0, index=128)
        """
        if axis == 0:
            return self.data[index, :, :]
        elif axis == 1:
            return self.data[:, index, :]
        elif axis == 2:
            return self.data[:, :, index]
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, or 2.")

    def get_data_range(self) -> Tuple[float, float]:
        """
        Get the minimum and maximum intensity values.

        Returns
        -------
        tuple
            (min_value, max_value)

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> vmin, vmax = img.get_data_range()
        """
        return float(self.data.min()), float(self.data.max())

    def copy(self) -> "MRIImage":
        """
        Create a deep copy of the image.

        Returns
        -------
        MRIImage
            A new image object with copied data.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> img_copy = img.copy()
        """
        return MRIImage(
            data=self.data.copy(),
            affine=self.affine.copy(),
            header=self.header.copy() if self.header else None,
            filepath=self.filepath,
        )

    def apply_mask(self, mask: np.ndarray, fill_value: float = 0.0) -> "MRIImage":
        """
        Apply a binary mask to the image.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask (same shape as image data).
        fill_value : float, optional
            Value to use for masked regions. Default is 0.

        Returns
        -------
        MRIImage
            New image with mask applied.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> mask = MRIImage.load('brain_mask.nii.gz').data > 0
        >>> masked_img = img.apply_mask(mask)
        """
        if mask.shape != self.data.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {self.data.shape}"
            )

        new_data = self.data.copy()
        new_data[~mask] = fill_value

        return MRIImage(new_data, self.affine.copy(), self.header, self.filepath)

    def __repr__(self) -> str:
        """String representation of the image."""
        info = f"MRIImage(shape={self.shape}, dtype={self.data.dtype}"
        if self.filepath:
            info += f", file='{self.filepath.name}'"
        info += ")"
        return info

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()
