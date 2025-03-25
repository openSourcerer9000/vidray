#!/usr/bin/env python3
"""
vidray.py

A lazy video processing library using xarray and Dask distributed. This module
defines two main classes, Img and Vid, which represent a single image frame and
a collection of frames (video), respectively. Operations (addition, subtraction,
multiplication, division) are implemented via a generic binary operation helper
and remain lazy until a preview or write is triggered.
"""

from pathlib import Path
from typing import List, Optional, Union
import operator
import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed
import imageio
from PIL import Image
import pandas as pd

# We import the Scalable mixin from transforms.py
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from parallelize_it import Cliente
    from transformz import Scalable, RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS
else:
    # uses current package visibility
    from .parallelize_it import Cliente
    from .transformz import Scalable, RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS

#####################################
# 1) Utility for arithmetic ops
#####################################
def _parallelBinaryOp(xr1: xr.DataArray,
                      xr2: xr.DataArray,
                      op,
                      absResult: bool = False,
                      intermediate=np.float32) -> xr.DataArray:
    """
    Generic helper to apply a binary operator in parallel using xarray + Dask.
    """
    import numpy as np
    result = xr.apply_ufunc(
        op,
        xr1.astype(intermediate),
        xr2.astype(intermediate),
        dask="parallelized",
        output_dtypes=[intermediate]
    )
    if absResult:
        result = xr.apply_ufunc(np.abs, result, dask="parallelized", output_dtypes=[intermediate])
    # Clip to [0, 255] for typical 8-bit image ops
    return result.clip(0, 255).astype(np.uint8)

#####################################
# 2) Mixin for arithmetic
#####################################
class BinaryOperable:
    """
    Mixin that provides arithmetic operator support with other Img/Vid objects or scalars.
    """

    def _applyBinaryOp(self,
                       other: Union["BinaryOperable", int, float],
                       op,
                       absResult: bool = False,
                       intermediate=np.float32):
        """
        Applies a binary operator between self.data and other (which may be an Img/Vid or a scalar).
        """
        if isinstance(other, (int, float)):
            # For scalars, xarray will broadcast automatically if we wrap in a zero-D DataArray
            otherData = xr.DataArray(other)
        else:
            otherData = other.data

        newData = _parallelBinaryOp(
            self.data,
            otherData,
            op,
            absResult=absResult,
            intermediate=intermediate
        )
        return self.__class__.fromData(newData)

#####################################
# 3) Img class
#####################################
class Img(BinaryOperable, Scalable):
    """
    Single-frame image class backed by a lazy xarray.DataArray.

    Example::
        >>> im = Img(pth=Path("example.png"))
    """
    def __init__(
        self,
        pth: Optional[Path] = None,
        dataArr: Optional[xr.DataArray] = None,
        chunks: tuple = (512, 512, -1)
    ):
        if dataArr is not None:
            self.data = dataArr
            return
        if pth is None:
            raise ValueError("Either 'pth' or 'dataArr' must be provided for Img.")

        # Read one frame to get shape and dtype
        example = imageio.imread(str(pth))
        self.shape = example.shape
        self.dtype = example.dtype
        lazyRead = delayed(imageio.imread)(str(pth))
        darr = da.from_delayed(lazyRead, shape=self.shape, dtype=self.dtype)
        self.data = xr.DataArray(darr, dims=["y", "x", "channel"]).chunk(chunks)

    @classmethod
    def fromData(cls, dataArr: xr.DataArray) -> "Img":
        """
        Creates an Img instance from an existing xarray.DataArray.
        """
        obj = cls.__new__(cls)
        obj.data = dataArr
        return obj

    def compute(self) -> xr.DataArray:
        """
        Computes the underlying DataArray.
        """
        return self.data.compute()

    def toNumpy(self) -> np.ndarray:
        """
        Returns the computed image as a NumPy array.
        """
        return self.compute().values

    # Arithmetic ops
    def __add__(self, other) -> "Img":
        return self._applyBinaryOp(other, operator.add)

    def __sub__(self, other) -> "Img":
        return self._applyBinaryOp(other, operator.sub, absResult=True)

    def __mul__(self, other) -> "Img":
        return self._applyBinaryOp(other, operator.mul)

    def __truediv__(self, other) -> "Img":
        return self._applyBinaryOp(other, operator.truediv, intermediate=np.float32)

#####################################
# 4) Vid class
#####################################
class Vid(BinaryOperable, Scalable):
    """
    Video-like collection of frames stored as a lazy xarray.DataArray with dims
    ["frame", "y", "x", "channel"].
    """

    def __init__(self, pathsOrImgs: Union[List[Path], List[Img], Path], chunks: tuple = (1, 512, 512, -1)):
        # Possibly accept a single Path or a list of them
        if isinstance(pathsOrImgs, Path):
            if pathsOrImgs.is_dir():
                pngs = list(pathsOrImgs.glob("*.png"))
                if not pngs:
                    files = list(pathsOrImgs.glob('*.*'))
                    suffs = pd.Series([f.suffix for f in files])
                    uniq = set(suffs.unique())
                    imgtyps = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
                    if not imgtyps.intersection(uniq):
                        raise ValueError(f"No PNG or other image files found in directory. Accepted types: {imgtyps}")
                    suffs = suffs[suffs.isin(imgtyps)]
                    suff = suffs.value_counts().index[0]
                    pathsOrImgs = list(pathsOrImgs.glob(f"*{suff}"))
                else:
                    pathsOrImgs = pngs
            else:
                pathsOrImgs = [pathsOrImgs]

        if not pathsOrImgs:
            raise ValueError("No frames provided to Vid.")

        first = pathsOrImgs[0]
        if isinstance(first, Path):
            # Lazy read from multiple PNG paths
            example = imageio.imread(str(first))
            frameShape = example.shape
            frameDtype = example.dtype
            daList = [
                da.from_delayed(delayed(imageio.imread)(str(p)), shape=frameShape, dtype=frameDtype)
                for p in pathsOrImgs
            ]
            stacked = da.stack(daList, axis=0)
            self.data = xr.DataArray(stacked, dims=["frame", "y", "x", "channel"]).chunk(chunks)
        elif isinstance(first, Img):
            # Stacking Img objects
            dataList = [img.data.expand_dims(dim={"frame": [i]}) for i, img in enumerate(pathsOrImgs)]
            self.data = xr.concat(dataList, dim="frame").chunk(chunks)
        else:
            raise TypeError("pathsOrImgs must be a Path or list of Path/Img objects.")

    @classmethod
    def fromData(cls, dataArr: xr.DataArray) -> "Vid":
        """
        Creates a Vid instance from an existing DataArray with dims ["frame", "y", "x", "channel"].
        """
        obj = cls.__new__(cls)
        obj.data = dataArr
        return obj

    def frame(self, index: int = 0) -> Image:
        """
        Returns a single frame as a PIL Image (converting the computed array).
        """
        # Persist Intermediate Results:
        # If the transformation graph is very deep or fragmented, consider calling .persist() on the scaled video before preview.
        # Select and persist one frame
        lazy_frame = self.data.isel(frame=index).persist()
        arr = lazy_frame.compute().values
        return Image.fromarray(arr)

        return arr
    def preview(self, index: int = 0) -> Image:
        """
        Returns a single frame as a NumPy array (converting the computed array).
        """
        return Image.fromarray(self.frame(index))

    def writePngs(self, outDest: Union[Path, list], pre: str = "frame", overwrite: bool = False):
        """
        Writes all frames as PNG images in parallel.

        If outDest is a Path (directory), filenames are generated as:
        pre + {frame:08d}.png
        within that directory.
        If outDest is a list of Paths, these are used directly.
        """
        from dask import delayed
        nFrames = self.data.sizes["frame"]
        if isinstance(outDest, Path):
            outDest.mkdir(parents=True, exist_ok=True)
            filePaths = [outDest / f"{pre}{i:08d}.png" for i in range(nFrames)]
        elif isinstance(outDest, list):
            filePaths = outDest
            if len(filePaths) != nFrames:
                raise ValueError(f"Expected {nFrames} file paths, got {len(filePaths)}")
        else:
            raise TypeError("outDest must be a Path or a list of Paths")

        def writeFrame(npFrame: np.ndarray, outPth: Path) -> str:
            if outPth.exists() and not overwrite:
                raise FileExistsError(f"File {outPth} already exists and overwrite=False.")
            imageio.imwrite(str(outPth), npFrame)
            return f"Saved {outPth.name}"

        tasks = []
        for i in range(nFrames):
            frameDA = self.data.isel(frame=i)
            frameDelayed = delayed(frameDA.compute)()
            tasks.append(delayed(writeFrame)(frameDelayed, filePaths[i]))

        from dask.distributed import Client, progress
        futures = Client.current().compute(tasks)
        print("Writing PNGs in parallel; check Dask dashboard for progress.")
        progress(futures)
        results = Client.current().gather(futures)
        return results

    def visualizeGraph(self, filename: str = "vid_graph.png") -> None:
        """
        Visualizes the underlying Dask task graph and saves it to a file.
        """
        self.data.data.visualize(filename=filename)
        print(f"Dask graph saved to {filename}")

    def compute(self) -> xr.DataArray:
        """
        Computes the entire video dataset.
        """
        return self.data.compute()

    # Arithmetic ops
    def __add__(self, other) -> "Vid":
        return self._applyBinaryOp(other, operator.add)

    def __sub__(self, other) -> "Vid":
        return self._applyBinaryOp(other, operator.sub, absResult=True)

    def __mul__(self, other) -> "Vid":
        return self._applyBinaryOp(other, operator.mul)

    def __truediv__(self, other) -> "Vid":
        return self._applyBinaryOp(other, operator.truediv, intermediate=np.float32)
