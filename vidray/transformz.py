#!/usr/bin/env python3
"""
transformz.py

Scalable mixin that supports scaling an image/video (xarray.DataArray) using one of three strategies.
Pivot points are specified as the point in both the scaled image and canvas that should be aligned.
Example:
    >>> scaled = obj.scale(scaleFactor=0.5, mode=RESIZE_IMAGE, pvt={"top":0,"left":0})
    >>> scaled.compute().values
"""

import numpy as np
import xarray as xr

RESIZE_BOTH   = 0
RESIZE_IMAGE  = 1
RESIZE_CANVAS = 2

class Scalable:
    def scale(self, scaleFactor: float = None, width: int = None, height: int = None,
              mode: int = RESIZE_BOTH, pvt: dict = None, interpolation: str = "cubic") -> "Scalable":
        data = self.data
        if "y" not in data.dims or "x" not in data.dims:
            raise ValueError("Data must have 'y' and 'x' dims")
        origH, origW = data.sizes["y"], data.sizes["x"]
        if scaleFactor is not None:
            newH, newW = int(round(origH * scaleFactor)), int(round(origW * scaleFactor))
        elif width is not None or height is not None:
            if width is None:
                newH, newW = height, int(round(height * (origW/origH)))
            elif height is None:
                newW, newH = width, int(round(width * (origH/origW)))
            else:
                newW, newH = width, height
        else:
            raise ValueError("Must provide scaleFactor or width/height")
        pvt = pvt or {"top": 0, "left": 0}
        # List-of-strategy lambdas; each returns a new DataArray.
        strategies = [
            lambda: self._scaleFull(data, newH, newW, interpolation),
            lambda: self._scaleImageOnly(data, newH, newW, self._computeAnchorFromPivot(pvt, newH, newW, origH, origW), interpolation),
            lambda: self._scaleCanvasOnly(data, newH, newW, self._computeAnchorFromPivot(pvt, origH, origW, newH, newW))
        ]
        newData = strategies[mode]()
        return self.__class__.fromData(newData)
    
    @staticmethod
    def _scaleFull(data: xr.DataArray, newH: int, newW: int, interp: str) -> xr.DataArray:
        origH, origW = data.sizes["y"], data.sizes["x"]
        yNew = np.linspace(0, origH - 1, newH)
        xNew = np.linspace(0, origW - 1, newW)
        return data.interp(y=yNew, x=xNew, method=interp).clip(0,255).astype(np.uint8)
    
    @staticmethod
    def _scaleImageOnly(data: xr.DataArray, newH: int, newW: int, anchor: tuple[int, int],
                        interp: str) -> xr.DataArray:
        scaled = Scalable._scaleFull(data, newH, newW, interp)
        blank = data.copy(deep=False)
        blank[:] = 0
        return Scalable._paste(blank, scaled, anchor, clip=True)
    
    # @staticmethod
    # def _scaleCanvasOnly(data: xr.DataArray, newH: int, newW: int, anchor: tuple[int, int]) -> xr.DataArray:
    #     dims = data.dims
    #     coords = {d: data.coords.get(d, np.arange(data.sizes[d])) for d in dims}
    #     shape = [newH if d=="y" else newW if d=="x" else data.sizes[d] for d in dims]
    #     blank = xr.DataArray(np.zeros(shape, dtype=data.dtype), dims=dims, coords=coords)
    #     return Scalable._paste(blank, data, anchor, clip=True)
    @staticmethod
    def _scaleCanvasOnly(data: xr.DataArray, newH: int, newW: int, anchor: tuple[int, int]) -> xr.DataArray:
        """
        Returns a new DataArray with dims updated to newH/newW for 'y'/'x'
        and with all coordinates generated from np.arange.
        """
        dims = data.dims  # e.g. ('frame', 'y', 'x', 'channel')
        # Build new shape and coordinates: for "y" and "x" use newH/newW, else use original size.
        shape = []
        coords = {}
        for d in dims:
            if d == "y":
                shape.append(newH)
                coords[d] = np.arange(newH)
            elif d == "x":
                shape.append(newW)
                coords[d] = np.arange(newW)
            else:
                size = data.sizes[d]
                shape.append(size)
                coords[d] = np.arange(size)
        blankArr = np.zeros(shape, dtype=data.dtype)
        blank = xr.DataArray(blankArr, dims=dims, coords=coords)
        return Scalable._paste(blank, data, anchor, clip=True)


    @staticmethod
    def _paste(blank: xr.DataArray, src: xr.DataArray, anchor: tuple[int, int], clip: bool = False) -> xr.DataArray:
        newData = blank.copy(deep=False)
        ay, ax = anchor
        h, w = src.sizes["y"], src.sizes["x"]
        BH, BW = blank.sizes["y"], blank.sizes["x"]
        y1, y2 = ay, ay + h
        x1, x2 = ax, ax + w
        if clip:
            srcY1, srcY2 = 0, h
            if y1 < 0: srcY1, y1 = -y1, 0
            if y2 > BH: srcY2, y2 = srcY2 - (y2 - BH), BH
            srcX1, srcX2 = 0, w
            if x1 < 0: srcX1, x1 = -x1, 0
            if x2 > BW: srcX2, x2 = srcX2 - (x2 - BW), BW
        newData[dict(y=slice(y1, y2), x=slice(x1, x2))] = src[dict(y=slice(srcY1, srcY2), x=slice(srcX1, srcX2))]
        return newData

    @staticmethod
    def _computeAnchorFromPivot(pvt: dict, scaledH: int, scaledW: int,
                                canvasH: int, canvasW: int) -> tuple[int, int]:
        # Interpret pivot keys as the pivot point in each image.
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
        # For vertical axis, use "bottom" if available; otherwise "top".
        spY = scaledH - getPivot("bottom", scaledH) if "bottom" in pvt else getPivot("top", scaledH)
        cpY = canvasH - getPivot("bottom", canvasH) if "bottom" in pvt else getPivot("top", canvasH)
        # For horizontal axis, use "right" if available; otherwise "left".
        spX = scaledW - getPivot("right", scaledW) if "right" in pvt else getPivot("left", scaledW)
        cpX = canvasW - getPivot("right", canvasW) if "right" in pvt else getPivot("left", canvasW)
        return int(round(cpY - spY)), int(round(cpX - spX))
