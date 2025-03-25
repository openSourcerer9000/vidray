#!/usr/bin/env python3
"""
transformz.py

Scalable mixin that supports scaling (and other transforms) on an xarray.DataArray.
It supports three strategies:
  - RESIZE_BOTH: scales the entire image (lazy via interpolation).
  - RESIZE_IMAGE: scales the image then pastes it into a blank canvas matching the original shape.
  - RESIZE_CANVAS: creates a new canvas of specified width/height and pastes the original image onto it.
  
For RESIZE_CANVAS, the transformation is done eagerly (using np.pad and concatenation)
to add zero‐padding on the top, bottom, left, and/or right as needed.
  
Constants:
  RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS
"""

import numpy as np
import xarray as xr
import dask.array as da
from PIL import Image

RESIZE_BOTH   = 0
RESIZE_IMAGE  = 1
RESIZE_CANVAS = 2

class Scalable:
    def scale(self, scaleFactor: float = None, width: int = None, height: int = None,
              mode: int = RESIZE_BOTH, pvt: dict = None, interpolation: str = "cubic") -> "Scalable":
        """
        Lazily scales self.data according to one of three strategies.
        Nothing is computed until you trigger compute()/preview()/write.
        
        For RESIZE_CANVAS mode on 3D data, the image is computed and then padded with zeros.
        """
        data = self.data
        if "y" not in data.dims or "x" not in data.dims:
            raise ValueError("Data must have 'y' and 'x' dims")
        origH, origW = data.sizes["y"], data.sizes["x"]
        if scaleFactor is not None:
            newH = int(round(origH * scaleFactor))
            newW = int(round(origW * scaleFactor))
        elif width is not None or height is not None:
            if width is None:
                newH = height
                newW = int(round(height * (origW / origH)))
            elif height is None:
                newW = width
                newH = int(round(width * (origH / origW)))
            else:
                newW, newH = width, height
        else:
            raise ValueError("Must provide scaleFactor or width/height")
        pvt = pvt or {"top": 0, "left": 0}
        
        if mode == RESIZE_BOTH:
            newData = self._scaleFull(data, newH, newW, interpolation)
            return self.__class__.fromData(newData)
        elif mode == RESIZE_IMAGE:
            anchor = self._computeAnchorFromPivot(pvt, newH, newW, origH, origW)
            newData = self._scaleImageOnly(data, newH, newW, anchor, interpolation)
            return self.__class__.fromData(newData)
        elif mode == RESIZE_CANVAS:
            # In RESIZE_CANVAS mode we compute the underlying data and then pad it.
            newData = self._scaleCanvasOnly(data, newH, newW, pvt)
            return self.__class__.fromData(newData)
        else:
            raise ValueError("Invalid mode")

    def _scaleFull(self, data: xr.DataArray, newH: int, newW: int, interp: str) -> xr.DataArray:
        yNew = np.linspace(0, data.sizes["y"] - 1, newH)
        xNew = np.linspace(0, data.sizes["x"] - 1, newW)
        return data.interp(y=yNew, x=xNew, method=interp).clip(0, 255).astype(np.uint8)

    def _scaleImageOnly(self, data: xr.DataArray, newH: int, newW: int, anchor: tuple, interp: str) -> xr.DataArray:
        scaled = self._scaleFull(data, newH, newW, interp)
        blank = data.copy(deep=True)
        blank[:] = 0
        return self._paste(blank, data, anchor, clip=True)
    
    def _scaleCanvasOnly(self, data: xr.DataArray, newH: int, newW: int, pvt: dict) -> xr.DataArray:
        """
        Eagerly computes data and adjusts it to a new canvas of shape (newH, newW, channel)
        by either padding with zeros (if new size is larger) or cropping (if smaller).
        The pivot dictionary (pvt) determines the alignment:
          - If "bottom" is in pvt, the bottom of the image is preserved (crop from top or pad on top).
          - Otherwise, the top is preserved.
          - Similarly for "right" (preserve right) versus left.
        """
        # Compute the underlying image as a NumPy array.
        arr = data.compute().values  # shape: (origH, origW, channel)
        origH, origW = arr.shape[:2]

        def adjust_dim(arr, new_size, axis, pivot):
            orig_size = arr.shape[axis]
            if new_size == orig_size:
                return arr
            elif new_size < orig_size:
                # Crop: if pivot specified (e.g., "bottom" for axis=0) preserve bottom, else top.
                if pivot is not None:
                    start = orig_size - new_size
                else:
                    start = 0
                slices = [slice(None)] * arr.ndim
                slices[axis] = slice(start, start + new_size)
                return arr[tuple(slices)]
            else:
                # Pad: if pivot specified, add padding before (e.g., "bottom" -> pad on top) to preserve bottom.
                pad = new_size - orig_size
                if pivot is not None:
                    if axis == 0:
                        pad_width = ((pad, 0), (0, 0), (0, 0))
                    elif axis == 1:
                        pad_width = ((0, 0), (pad, 0), (0, 0))
                else:
                    if axis == 0:
                        pad_width = ((0, pad), (0, 0), (0, 0))
                    elif axis == 1:
                        pad_width = ((0, 0), (0, pad), (0, 0))
                return np.pad(arr, pad_width, mode='constant', constant_values=0)

        # Adjust vertical dimension.
        arr = adjust_dim(arr, newH, axis=0, pivot="bottom" if "bottom" in pvt else None)
        # Adjust horizontal dimension.
        arr = adjust_dim(arr, newW, axis=1, pivot="right" if "right" in pvt else None)

        new_coords = {
            "y": np.arange(newH),
            "x": np.arange(newW),
            "channel": np.arange(arr.shape[2])
        }
        return xr.DataArray(arr, dims=("y", "x", "channel"), coords=new_coords)

    def _computeAnchorFromPivot(self, pvt: dict, scaledH: int, scaledW: int, origH: int, origW: int) -> tuple:
        spY = scaledH - (float(pvt.get("bottom", 0)) if "bottom" in pvt else float(pvt.get("top", 0)))
        cpY = origH - (float(pvt.get("bottom", 0)) if "bottom" in pvt else float(pvt.get("top", 0)))
        spX = scaledW - (float(pvt.get("right", 0)) if "right" in pvt else float(pvt.get("left", 0)))
        cpX = origW - (float(pvt.get("right", 0)) if "right" in pvt else float(pvt.get("left", 0)))
        return (int(round(cpY - spY)), int(round(cpX - spX)))

    def _paste(self, blank: xr.DataArray, src: xr.DataArray, anchor: tuple, clip: bool = False) -> xr.DataArray:
        ay, ax = anchor
        blank_simple = blank.assign_coords({dim: np.arange(blank.sizes[dim]) for dim in blank.dims})
        src_simple = (src.assign_coords({dim: np.arange(src.sizes[dim]) for dim in src.dims})
                          .rename({'y': 'y_src', 'x': 'x_src'}))
        pasted = xr.apply_ufunc(
            lambda b, s, ay, ax, clip: Scalable._paste_func(b, s, ay, ax, clip),
            blank_simple, src_simple,
            input_core_dims=[["y", "x", "channel"], ["y_src", "x_src", "channel"]],
            output_core_dims=[["y", "x", "channel"]],
            kwargs={"ay": ay, "ax": ax, "clip": clip},
            vectorize=False,
            dask="parallelized",
            output_dtypes=[blank.dtype],
            join="override",
            dask_gufunc_kwargs={"allow_rechunk": True}
        )
        return pasted

    @staticmethod
    def _paste_func(blank, src, ay, ax, clip):
        blank = blank.copy()
        h, w = src.shape[0], src.shape[1]
        BH, BW = blank.shape[0], blank.shape[1]
        y1, y2 = ay, ay + h
        x1, x2 = ax, ax + w
        srcY1, srcY2 = 0, h
        srcX1, srcX2 = 0, w
        if clip:
            if y1 < 0:
                srcY1, y1 = -y1, 0
            if y2 > BH:
                srcY2, y2 = h - (y2 - BH), BH
            if x1 < 0:
                srcX1, x1 = -x1, 0
            if x2 > BW:
                srcX2, x2 = w - (x2 - BW), BW
        blank[y1:y2, x1:x2, ...] = src[srcY1:srcY2, srcX1:srcX2, ...]
        return blank
