#!/usr/bin/env python3
"""
vidray.py

A lazy video processing library using xarray and Dask distributed.
Defines Img and Vid classes. Operations remain lazy until preview() or writePngs() is triggered.
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

if __package__ is None or __package__ == "":
    from parallelize_it import Cliente
    from transformz import Scalable, RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS
else:
    from .parallelize_it import Cliente
    from .transformz import Scalable, RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS

def _parallelBinaryOp(xr1: xr.DataArray,
                      xr2: xr.DataArray,
                      op,
                      absResult: bool = False,
                      intermediate=np.float32) -> xr.DataArray:
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
    return result.clip(0, 255).astype(np.uint8)

class BinaryOperable:
    def _applyBinaryOp(self,
                       other: Union["BinaryOperable", int, float],
                       op,
                       absResult: bool = False,
                       intermediate=np.float32):
        if isinstance(other, (int, float)):
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

class Img(BinaryOperable, Scalable):
    def __init__(self, pth: Optional[Path] = None, dataArr: Optional[xr.DataArray] = None, chunks: tuple = (512, 512, -1)):
        if dataArr is not None:
            self.data = dataArr
            return
        if pth is None:
            raise ValueError("Either 'pth' or 'dataArr' must be provided for Img.")
        example = imageio.imread(str(pth))
        self.shape = example.shape
        self.dtype = example.dtype
        lazyRead = delayed(imageio.imread)(str(pth))
        darr = da.from_delayed(lazyRead, shape=self.shape, dtype=self.dtype)
        self.data = xr.DataArray(darr, dims=["y", "x", "channel"]).chunk(chunks)
    @classmethod
    def fromData(cls, dataArr: xr.DataArray) -> "Img":
        obj = cls.__new__(cls)
        obj.data = dataArr
        return obj
    def compute(self) -> xr.DataArray:
        return self.data.compute()
    def toNumpy(self) -> np.ndarray:
        return self.compute().values
    def __add__(self, other) -> "Img":
        return self._applyBinaryOp(other, operator.add)
    def __sub__(self, other) -> "Img":
        return self._applyBinaryOp(other, operator.sub, absResult=True)
    def __mul__(self, other) -> "Img":
        return self._applyBinaryOp(other, operator.mul)
    def __truediv__(self, other) -> "Img":
        return self._applyBinaryOp(other, operator.truediv, intermediate=np.float32)

class Vid(BinaryOperable, Scalable):
    def __init__(self, pathsOrImgs: Union[List[Path], List[Img], Path],suff='.jpg', chunks: tuple = (1, 512, 512, -1)):
        if isinstance(pathsOrImgs, Path):
            if pathsOrImgs.is_dir():
                pngs = list(pathsOrImgs.glob(f"*{suff}"))
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
            example = imageio.imread(str(first))
            frameShape = example.shape
            frameDtype = example.dtype
            daList = [da.from_delayed(delayed(imageio.imread)(str(p)), shape=frameShape, dtype=frameDtype)
                      for p in pathsOrImgs]
            stacked = da.stack(daList, axis=0)
            self.data = xr.DataArray(stacked, dims=["frame", "y", "x", "channel"]).chunk(chunks)
        elif isinstance(first, Img):
            dataList = [img.data.expand_dims(dim={"frame": [i]}) for i, img in enumerate(pathsOrImgs)]
            self.data = xr.concat(dataList, dim="frame").chunk(chunks)
        else:
            raise TypeError("pathsOrImgs must be a Path or list of Path/Img objects.")
    @classmethod
    def fromData(cls, dataArr: xr.DataArray) -> "Vid":
        obj = cls.__new__(cls)
        obj.data = dataArr
        return obj
    def frame(self, index: int = 0) -> Image:
        # If canvas mode was selected, process the selected frame lazily using _paste_frame.
        if hasattr(self, "_scaling_mode") and self._scaling_mode == RESIZE_CANVAS:
            params = self._canvas_params
            frame_da = self.data.isel(frame=index)
            transformed = Scalable._paste_frame(frame_da, params["newH"], params["newW"], params["pvt"])
            arr = transformed.compute().values
            return Image.fromarray(arr)
        else:
            arr = self.data.isel(frame=index).compute().values
            return Image.fromarray(arr)
    def preview(self, index: int = 0) -> Image:
        return self.frame(index)
    def bounce(self, outDest: Union[Path, list], pre: str = "frame", overwrite: bool = True,suff='.jpg'):
        from dask import delayed
        nFrames = self.data.sizes["frame"]
        if isinstance(outDest, Path):
            outDest.mkdir(parents=True, exist_ok=True)
            filePaths = [outDest / f"{pre}{i:08d}{suff}" for i in range(nFrames)]
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
            if hasattr(self, "_scaling_mode") and self._scaling_mode == RESIZE_CANVAS:
                params = self._canvas_params
                frameDA = Scalable._paste_frame(self.data.isel(frame=i), params["newH"], params["newW"], params["pvt"])
            else:
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
        self.data.data.visualize(filename=filename)
        print(f"Dask graph saved to {filename}")
    def compute(self) -> xr.DataArray:
        return self.data.compute()
    def __add__(self, other) -> "Vid":
        return self._applyBinaryOp(other, operator.add)
    def __sub__(self, other) -> "Vid":
        return self._applyBinaryOp(other, operator.sub, absResult=True)
    def __mul__(self, other) -> "Vid":
        return self._applyBinaryOp(other, operator.mul)
    def __truediv__(self, other) -> "Vid":
        return self._applyBinaryOp(other, operator.truediv, intermediate=np.float32)

import os
import shutil
def savevid(frmpth,outvid,suff='.jpg'):
    opth = frmpth.parent / 'temp_vid_frames'
    shutil.rmtree(opth,ignore_errors=True)
    opth.mkdir(exist_ok=True)
    os.chdir(str(frmpth.parent))
    frms = sorted(list(frmpth.glob('*'+suff)))

    fnms = [f'frame{str(i).zfill(5)}.png' for i in range(len(frms)) ]
    [shutil.copy(frm, opth/fn) for frm,fn in zip(frms,fnms)]
    Path(outvid).unlink(missing_ok=True)
    ! C:\Py\ffmpeg\bin\ffmpeg -framerate 30 -i {opth.name}/frame%05d.png -c:v libx264 -pix_fmt yuv420p {outvid}
    shutil.rmtree(opth,ignore_errors=True)
    print(f'Saved video: {outvid}')