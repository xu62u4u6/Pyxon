"""
Core visualization utilities for MRI images.

This module provides visualization functions for MRI data, including
slice viewing, overlays, and colormap utilities.
"""

from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches


def show_slice(
    data: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    cmap: str = 'gray',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    colorbar: bool = False,
    ax: Optional[plt.Axes] = None,
    norm: Optional[object] = None,
) -> plt.Axes:
    """
    Display a 2D slice from a 3D volume.

    Parameters
    ----------
    data : np.ndarray
        3D image data.
    slice_idx : int, optional
        Index of the slice to display. If None, uses middle slice.
    axis : int, optional
        Axis along which to slice (0, 1, or 2). Default is 2 (axial).
    cmap : str, optional
        Colormap name. Default is 'gray'.
    vmin, vmax : float, optional
        Data range for colormap. If None, uses data min/max.
    title : str, optional
        Plot title.
    colorbar : bool, optional
        Whether to show colorbar. Default is False.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> from pyxon.visualization import show_slice
    >>> show_slice(data, slice_idx=128, axis=2)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get slice
    if slice_idx is None:
        slice_idx = data.shape[axis] // 2
    
    if axis == 0:
        slice_data = data[slice_idx, :, :]
    elif axis == 1:
        slice_data = data[:, slice_idx, :]
    elif axis == 2:
        slice_data = data[:, :, slice_idx]
    else:
        raise ValueError(f"Invalid axis: {axis}")
    
    # Plot
    im = ax.imshow(slice_data.T, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, 
                   origin='lower', interpolation='nearest')
    
    if title:
        ax.set_title(title)
    else:
        axis_names = ['Sagittal', 'Coronal', 'Axial']
        ax.set_title(f'{axis_names[axis]} slice {slice_idx}')
    
    ax.axis('off')
    
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    return ax


def show_orthogonal(
    data: np.ndarray,
    coords: Optional[Tuple[int, int, int]] = None,
    cmap: str = 'gray',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    norm: Optional[object] = None,
) -> plt.Figure:
    """
    Show orthogonal views (sagittal, coronal, axial) of a 3D volume.

    Parameters
    ----------
    data : np.ndarray
        3D image data.
    coords : tuple of int, optional
        (x, y, z) coordinates for crosshair. If None, uses center.
    cmap : str, optional
        Colormap name. Default is 'gray'.
    vmin, vmax : float, optional
        Data range for colormap.
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.

    Examples
    --------
    >>> from pyxon.visualization import show_orthogonal
    >>> show_orthogonal(data, coords=(128, 128, 128))
    """
    if coords is None:
        coords = tuple(s // 2 for s in data.shape)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Sagittal (x)
    show_slice(data, coords[0], axis=0, cmap=cmap, vmin=vmin, vmax=vmax,
               norm=norm, ax=axes[0])
    
    # Coronal (y)
    show_slice(data, coords[1], axis=1, cmap=cmap, vmin=vmin, vmax=vmax,
               norm=norm, ax=axes[1])
    
    # Axial (z)
    show_slice(data, coords[2], axis=2, cmap=cmap, vmin=vmin, vmax=vmax,
               norm=norm, ax=axes[2])
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def show_overlay(
    background: np.ndarray,
    overlay: np.ndarray,
    slice_idx: Optional[int] = None,
    axis: int = 2,
    bg_cmap: str = 'gray',
    overlay_cmap: str = 'hot',
    alpha: float = 0.5,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Display an overlay of two images.

    Parameters
    ----------
    background : np.ndarray
        Background image (3D).
    overlay : np.ndarray
        Overlay image (3D), same shape as background.
    slice_idx : int, optional
        Index of the slice to display.
    axis : int, optional
        Axis along which to slice. Default is 2.
    bg_cmap : str, optional
        Colormap for background. Default is 'gray'.
    overlay_cmap : str, optional
        Colormap for overlay. Default is 'hot'.
    alpha : float, optional
        Transparency of overlay (0-1). Default is 0.5.
    vmin, vmax : float, optional
        Data range for overlay colormap.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object.

    Examples
    --------
    >>> from pyxon.visualization import show_overlay
    >>> show_overlay(brain, segmentation, slice_idx=128, alpha=0.3)
    """
    if background.shape != overlay.shape:
        raise ValueError("Background and overlay must have same shape")
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get slice
    if slice_idx is None:
        slice_idx = background.shape[axis] // 2
    
    if axis == 0:
        bg_slice = background[slice_idx, :, :]
        ov_slice = overlay[slice_idx, :, :]
    elif axis == 1:
        bg_slice = background[:, slice_idx, :]
        ov_slice = overlay[:, slice_idx, :]
    elif axis == 2:
        bg_slice = background[:, :, slice_idx]
        ov_slice = overlay[:, :, slice_idx]
    else:
        raise ValueError(f"Invalid axis: {axis}")
    
    # Plot background
    ax.imshow(bg_slice.T, cmap=bg_cmap, origin='lower', interpolation='nearest')
    
    # Plot overlay (mask zero values)
    overlay_masked = np.ma.masked_where(ov_slice == 0, ov_slice)
    ax.imshow(overlay_masked.T, cmap=overlay_cmap, alpha=alpha, 
              vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
    
    if title:
        ax.set_title(title)
    
    ax.axis('off')
    
    return ax
