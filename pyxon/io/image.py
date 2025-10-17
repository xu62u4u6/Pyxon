"""
Object-oriented MRI image interface.

This module provides a high-level object-oriented API for working with MRI images,
built on top of the low-level I/O functions.
"""

from pathlib import Path
from typing import Optional, Tuple, Union, TYPE_CHECKING

import nibabel as nib
import numpy as np

from ..io.image_loader import load_image, save_nifti, save_mgz
from ..visualization.color_manager import ColorManager

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# Global ColorManager instance for this module
_color_manager = ColorManager()


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
    image_type : str, optional
        Type of image: 'anatomical', 'segmentation', 'binary_mask', 'value_map'.
        Default is 'anatomical'.

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
    image_type : str
        Type of the image.
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
    
    >>> # Load with specific type
    >>> seg = MRIImage.load_segmentation('seg.nii.gz')
    >>> mask = MRIImage.load_binary_mask('mask.nii.gz')
    
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
        image_type: str = "anatomical",
    ):
        self.data = data
        self.affine = affine
        self.header = header
        self.filepath = Path(filepath) if filepath else None
        self.image_type = image_type

    @classmethod
    def load(
        cls, 
        filepath: Union[str, Path],
        image_type: str = "anatomical",
    ) -> "MRIImage":
        """
        Load an MRI image from file.

        Parameters
        ----------
        filepath : str or Path
            Path to the image file (.nii, .nii.gz, or .mgz).
        image_type : str, optional
            Type of image: 'anatomical', 'segmentation', 'binary_mask', 'value_map'.
            Default is 'anatomical'. This affects default visualization settings.

        Returns
        -------
        MRIImage
            Loaded image object.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> seg = MRIImage.load('seg.nii.gz', image_type='segmentation')
        >>> mask = MRIImage.load('mask.nii.gz', image_type='binary_mask')
        """
        data, affine, header = load_image(
            filepath, with_affine=True, with_header=True
        )
        return cls(data, affine, header, filepath, image_type)

    @classmethod
    def load_segmentation(cls, filepath: Union[str, Path]) -> "MRIImage":
        """
        Load a segmentation mask image.

        Convenience method for loading segmentation masks with appropriate
        default visualization settings.

        Parameters
        ----------
        filepath : str or Path
            Path to the segmentation file.

        Returns
        -------
        MRIImage
            Loaded segmentation image.

        Examples
        --------
        >>> seg = MRIImage.load_segmentation('aparc+aseg.mgz')
        >>> seg.show()  # Will use appropriate colormap for labels
        """
        return cls.load(filepath, image_type="segmentation")

    @classmethod
    def load_binary_mask(cls, filepath: Union[str, Path]) -> "MRIImage":
        """
        Load a binary mask image.

        Convenience method for loading binary masks (0/1 or boolean values).

        Parameters
        ----------
        filepath : str or Path
            Path to the binary mask file.

        Returns
        -------
        MRIImage
            Loaded binary mask image.

        Examples
        --------
        >>> mask = MRIImage.load_binary_mask('brain_mask.mgz')
        >>> masked = img.apply_mask(mask.data > 0)
        """
        return cls.load(filepath, image_type="binary_mask")

    @classmethod
    def load_value_map(cls, filepath: Union[str, Path]) -> "MRIImage":
        """
        Load a continuous value map (e.g., statistical maps, probability maps).

        Convenience method for loading continuous value images.

        Parameters
        ----------
        filepath : str or Path
            Path to the value map file.

        Returns
        -------
        MRIImage
            Loaded value map image.

        Examples
        --------
        >>> stat_map = MRIImage.load_value_map('thickness.mgz')
        >>> stat_map.show(colorbar=True)
        """
        return cls.load(filepath, image_type="value_map")

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
            image_type=self.image_type,
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
        >>> mask = MRIImage.load_binary_mask('brain_mask.nii.gz')
        >>> masked_img = img.apply_mask(mask.data > 0)
        """
        if mask.shape != self.data.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {self.data.shape}"
            )

        new_data = self.data.copy()
        new_data[~mask] = fill_value

        return MRIImage(
            new_data, 
            self.affine.copy(), 
            self.header, 
            self.filepath,
            self.image_type
        )

    def __repr__(self) -> str:
        """String representation of the image."""
        info = f"MRIImage(type='{self.image_type}', shape={self.shape}, dtype={self.data.dtype}"
        if self.filepath:
            info += f", file='{self.filepath.name}'"
        info += ")"
        return info

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()

    # =========================================================================
    # Visualization methods (convenience wrappers)
    # =========================================================================

    def show(
        self,
        slice_idx: Optional[int] = None,
        axis: int = 2,
        cmap: Optional[str] = None,
        title: Optional[str] = None,
        colorbar: bool = False,
        ax: Optional["plt.Axes"] = None,
    ) -> "plt.Axes":
        """
        Display a slice of the image.

        This is a convenience method that wraps visualization.show_slice().
        Automatically selects appropriate colormap based on image_type if not specified.

        Parameters
        ----------
        slice_idx : int, optional
            Index of the slice to display. If None, uses middle slice.
        axis : int, optional
            Axis along which to slice (0=sagittal, 1=coronal, 2=axial).
        cmap : str, optional
            Colormap name. If None, auto-selected based on image_type:
            - 'anatomical': 'gray'
            - 'segmentation': FreeSurfer LUT colormap (fallback to 'tab20')
            - 'binary_mask': 'gray'
            - 'value_map': 'viridis'
        title : str, optional
            Plot title. If None, auto-generated.
        colorbar : bool, optional
            Whether to show colorbar.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> img.show()  # Show middle axial slice with gray colormap
        
        >>> seg = MRIImage.load_segmentation('seg.nii.gz')
        >>> seg.show()  # Automatically uses FreeSurfer colormap
        
        >>> img.show(slice_idx=100, axis=0, cmap='hot')  # Custom settings
        """
        from ..visualization.image import show_slice
        
        # Initialize norm as None
        norm = None
        
        # Auto-select colormap based on image type
        if cmap is None:
            if self.image_type == 'segmentation':
                # Use FreeSurfer colormap for segmentation
                try:
                    # Get matplotlib colormap and normalization
                    cmap, norm = _color_manager.get_matplotlib_cmap('freesurfer')
                except ValueError:
                    # Fallback to tab20 if FreeSurfer LUT not available
                    cmap = 'tab20'
                    norm = None
            else:
                cmap_defaults = {
                    'anatomical': 'gray',
                    'binary_mask': 'gray',
                    'value_map': 'viridis',
                }
                cmap = cmap_defaults.get(self.image_type, 'gray')
                norm = None
        
        # Auto-generate title
        if title is None:
            if self.filepath:
                title = f"{self.filepath.name} ({self.image_type})"
            else:
                title = f"MRIImage ({self.image_type})"
        
        return show_slice(
            self.data, slice_idx=slice_idx, axis=axis,
            cmap=cmap, title=title, colorbar=colorbar, norm=norm, ax=ax
        )

    def show_orthogonal(
        self,
        coords: Optional[Tuple[int, int, int]] = None,
        cmap: Optional[str] = None,
        title: Optional[str] = None,
    ) -> "plt.Figure":
        """
        Show orthogonal views (sagittal, coronal, axial).

        This is a convenience method that wraps visualization.show_orthogonal().
        Automatically selects appropriate colormap based on image_type if not specified.

        Parameters
        ----------
        coords : tuple of int, optional
            (x, y, z) coordinates for crosshair. If None, uses center.
        cmap : str, optional
            Colormap name. If None, auto-selected based on image_type:
            - 'anatomical': 'gray'
            - 'segmentation': FreeSurfer LUT colormap (fallback to 'tab20')
            - 'binary_mask': 'gray'
            - 'value_map': 'viridis'
        title : str, optional
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> img.show_orthogonal()  # Show center slices with auto colormap
        >>> seg = MRIImage.load_segmentation('seg.nii.gz')
        >>> seg.show_orthogonal()  # Automatically uses FreeSurfer colormap
        >>> img.show_orthogonal(coords=(100, 120, 90))
        """
        from ..visualization.image import show_orthogonal
        
        # Initialize norm as None
        norm = None
        
        # Auto-select colormap based on image type
        if cmap is None:
            if self.image_type == 'segmentation':
                # Use FreeSurfer colormap for segmentation
                try:
                    # Get matplotlib colormap and normalization
                    cmap, norm = _color_manager.get_matplotlib_cmap('freesurfer')
                except ValueError:
                    # Fallback to tab20 if FreeSurfer LUT not available
                    cmap = 'tab20'
                    norm = None
            else:
                cmap_defaults = {
                    'anatomical': 'gray',
                    'binary_mask': 'gray',
                    'value_map': 'viridis',
                }
                cmap = cmap_defaults.get(self.image_type, 'gray')
                norm = None
        
        if title is None and self.filepath:
            title = f"{self.filepath.name} ({self.image_type})"
        
        return show_orthogonal(self.data, coords=coords, cmap=cmap, title=title, norm=norm)

    def show_overlay(
        self,
        overlay: Union["MRIImage", np.ndarray],
        slice_idx: Optional[int] = None,
        axis: int = 2,
        overlay_cmap: str = 'hot',
        alpha: float = 0.5,
        title: Optional[str] = None,
    ) -> "plt.Axes":
        """
        Display this image with an overlay.

        This is a convenience method that wraps visualization.show_overlay().

        Parameters
        ----------
        overlay : MRIImage or np.ndarray
            Overlay image or data.
        slice_idx : int, optional
            Index of the slice to display.
        axis : int, optional
            Axis along which to slice.
        overlay_cmap : str, optional
            Colormap for overlay. Default is 'hot'.
        alpha : float, optional
            Transparency of overlay (0-1).
        title : str, optional
            Plot title.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object.

        Examples
        --------
        >>> img = MRIImage.load('brain.nii.gz')
        >>> seg = MRIImage.load('segmentation.nii.gz')
        >>> img.show_overlay(seg, alpha=0.3)
        """
        from ..visualization.image import show_overlay
        
        overlay_data = overlay.data if isinstance(overlay, MRIImage) else overlay
        
        if title is None and self.filepath:
            title = f"{self.filepath.name} with overlay"
        
        return show_overlay(
            self.data, overlay_data, slice_idx=slice_idx, axis=axis,
            overlay_cmap=overlay_cmap, alpha=alpha, title=title
        )
