#!/usr/bin/env python3
"""
transformz.py

Scalable mixin that supports lazy scaling of an image/video (xarray.DataArray)
using one of three strategies:
  - RESIZE_BOTH: scales the entire image.
  - RESIZE_IMAGE: scales the image then pastes it into a blank canvas of the original shape.
  - RESIZE_CANVAS: creates a new canvas (with new dimensions) and pastes the image onto it.

Pivot points are specified as the point in both the scaled image and canvas that should be aligned.
Nothing is computed until you load or write a frame.
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
        Nothing is computed until you call compute() (or load/write a frame).
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
        
        # Compute an anchor point for pasting.
        def getPivot(key, size):
            if key in pvt:
                try:
                    return float(pvt[key])
                except Exception:
                    s = str(pvt[key]).strip().lower()
                    if s.endswith("%"):
                        return float(s[:-1]) / 100.0 * size
                    if s.endswith("px"):
                        return float(s[:-2])
                    return float(s)
            return 0.0

        spY = newH - getPivot("bottom", newH) if "bottom" in pvt else getPivot("top", newH)
        cpY = origH - getPivot("bottom", origH) if "bottom" in pvt else getPivot("top", origH)
        spX = newW - getPivot("right", newW) if "right" in pvt else getPivot("left", newW)
        cpX = origW - getPivot("right", origW) if "right" in pvt else getPivot("left", origW)
        anchor = (int(round(cpY - spY)), int(round(cpX - spX)))
        
        # Choose the scaling strategy.
        if mode == RESIZE_BOTH:
            newData = self._scaleFull(data, newH, newW, interpolation)
        elif mode == RESIZE_IMAGE:
            newData = self._scaleImageOnly(data, newH, newW, anchor, interpolation)
        elif mode == RESIZE_CANVAS:
            newData = self._scaleCanvasOnly(data, newH, newW, anchor)
        else:
            raise ValueError("Invalid mode")
        
        # *** Ensure that if a "frame" dimension is present (e.g. for a video),
        # we rechunk so that each frame is its own chunk.
        if "frame" in newData.dims:
            newData = newData.chunk({"frame": 1})
        
        return self.__class__.fromData(newData)

    
    @staticmethod
    def _scale_func(arr, newH, newW, interp):
        """
        Scales a single image (NumPy array) using PIL.
        Wrapped by xr.apply_ufunc to remain lazy.
        """
        im = Image.fromarray(arr)
        if interp == "cubic":
            pil_interp = Image.BICUBIC
        elif interp == "linear":
            pil_interp = Image.BILINEAR
        elif interp == "nearest":
            pil_interp = Image.NEAREST
        else:
            pil_interp = Image.BICUBIC
        im_resized = im.resize((newW, newH), resample=pil_interp)
        return np.array(im_resized, dtype=np.uint8)
    
    @staticmethod
    def _scaleFull(data: xr.DataArray, newH: int, newW: int, interp: str) -> xr.DataArray:
        """
        Lazily scales the entire DataArray using xr.apply_ufunc.
        Assumes core dims ["y", "x", "channel"].
        """
        return xr.apply_ufunc(
            Scalable._scale_func,
            data,
            input_core_dims=[["y", "x", "channel"]],
            output_core_dims=[["y", "x", "channel"]],
            kwargs={"newH": newH, "newW": newW, "interp": interp},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.uint8],
            join="override"
        )
    
    @staticmethod
    def _paste_func(blank, src, ay, ax, clip):
        """
        Pastes src onto blank at anchor (ay, ax) on NumPy arrays.
        """
        blank = blank.copy()
        h, w = src.shape[0], src.shape[1]
        BH, BW = blank.shape[0], blank.shape[1]
        y1 = ay
        y2 = ay + h
        x1 = ax
        x2 = ax + w
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
    
    @staticmethod
    def _paste(blank: xr.DataArray, src: xr.DataArray, anchor: tuple, clip: bool = False) -> xr.DataArray:
        """
        Lazily pastes src onto blank using xr.apply_ufunc.
        This version forces the spatial dimensions ('y' and 'x') into a single chunk,
        so that the paste operation is applied to one large block rather than many tiny ones.
        """
        ay, ax = anchor

        # Force blank and src to have a single chunk along y and x.
        blank = blank.chunk({"y": blank.sizes["y"], "x": blank.sizes["x"]})
        src = src.chunk({"y": src.sizes["y"], "x": src.sizes["x"]})

        # Reassign coordinates to simple ranges.
        blank_simple = blank.assign_coords({dim: np.arange(blank.sizes[dim]) for dim in blank.dims})
        # For src, rename spatial dims so they don't conflict.
        src_simple = (src.assign_coords({dim: np.arange(src.sizes[dim]) for dim in src.dims})
                        .rename({'y': 'y_src', 'x': 'x_src'}))
        
        pasted = xr.apply_ufunc(
            Scalable._paste_func,
            blank_simple, src_simple,
            input_core_dims=[["y", "x", "channel"], ["y_src", "x_src", "channel"]],
            output_core_dims=[["y", "x", "channel"]],
            kwargs={"ay": ay, "ax": ax, "clip": clip},
            vectorize=False,  # operate on entire arrays at once
            dask="parallelized",
            output_dtypes=[blank.dtype],
            join="override",
            dask_gufunc_kwargs={"allow_rechunk": True}
        )
        return pasted


    
    @staticmethod
    def _scaleImageOnly(data: xr.DataArray, newH: int, newW: int, anchor: tuple, interp: str) -> xr.DataArray:
        """
        Lazily scales the image then pastes it onto a blank canvas matching the original shape.
        """
        scaled = Scalable._scaleFull(data, newH, newW, interp)
        blank_arr = da.zeros(data.shape, dtype=data.dtype)
        blank = xr.DataArray(blank_arr, dims=data.dims,
                             coords={dim: np.arange(data.sizes[dim]) for dim in data.dims})
        return Scalable._paste(blank, scaled, anchor, clip=True)
    
    @staticmethod
    def _scaleCanvasOnly(data: xr.DataArray, newH: int, newW: int, anchor: tuple) -> xr.DataArray:
        """
        Lazily creates a new canvas with dimensions newH/newW and pastes data onto it.
        (The output DataArray will have the new canvas dimensions.)
        """
        dims = list(data.dims)
        new_shape = []
        new_coords = {}
        for d in dims:
            if d == "y":
                new_shape.append(newH)
                new_coords[d] = np.arange(newH)
            elif d == "x":
                new_shape.append(newW)
                new_coords[d] = np.arange(newW)
            else:
                size = data.sizes[d]
                new_shape.append(size)
                new_coords[d] = np.arange(size)
        blank_arr = da.zeros(new_shape, dtype=data.dtype)
        blank = xr.DataArray(blank_arr, dims=dims, coords=new_coords)
        return Scalable._paste(blank, data, anchor, clip=True)
