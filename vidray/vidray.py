#!/usr/bin/env python3
"""
vidray.py

A lazy video processing library using xarray and Dask distributed.
Defines Img and Vid classes. Operations remain lazy until preview() or writePngs() is triggered.
"""
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from patchy import patch,unpatch
    from util import *
else:
    # uses current package visibility
    from .patchy import patch,unpatch
    from .util import *
from pathlib import Path
from typing import List, Optional, Union
import operator
import json
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

def _normalize_grid(z: Union[int, tuple]) -> tuple:
    if isinstance(z, int):
        return (z, z)
    if len(z) != 2:
        raise ValueError("Grid must be an int or a tuple of (rows, cols).")
    return tuple(z)

def _compute_patch_plan(h: int, w: int, z: Union[int, tuple], overlap: int) -> dict:
    zu, zv = _normalize_grid(z)
    if overlap < 0:
        raise ValueError("Overlap must be >= 0.")
    patch_h = ((h - overlap) // zu) + overlap
    patch_w = ((w - overlap) // zv) + overlap
    step_h = patch_h - overlap
    step_w = patch_w - overlap
    if step_h <= 0 or step_w <= 0:
        raise ValueError("Overlap is too large for the requested grid.")
    return {
        "grid": (zu, zv),
        "patch_size": (patch_h, patch_w),
        "step": (step_h, step_w),
    }

def _pad_spatial(darr: da.Array, overlap: int) -> da.Array:
    if overlap <= 0:
        return darr
    if darr.ndim == 4:
        pad_width = ((0, 0), (0, overlap), (0, overlap), (0, 0))
    elif darr.ndim == 3:
        pad_width = ((0, overlap), (0, overlap), (0, 0))
    else:
        raise ValueError("Expected 3D or 4D array for padding.")
    return da.pad(darr, pad_width=pad_width, mode="reflect")

def _write_vid_frames(vid: "Vid", outdir: Path, pre: str, suff: str, overwrite: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    n_frames = vid.data.sizes["frame"]
    for i in range(n_frames):
        out_pth = outdir / f"{pre}{i:08d}{suff}"
        if out_pth.exists() and not overwrite:
            raise FileExistsError(f"File {out_pth} already exists and overwrite=False.")
        frame = vid.data.isel(frame=i).compute().values
        imageio.imwrite(str(out_pth), frame)

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
                pngs = [p for p in pngs if not p.name.startswith('._')]
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
    def __len__(self) -> int:
        return self.data.sizes["frame"]
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

    def patch(self,
              z: Union[int, tuple],
              outdir: Optional[Path] = None,
              overlap: int = 20,
              tile_prefix: str = "tile_",
              frame_prefix: str = "frame",
              suff: str = ".jpg",
              overwrite: bool = True) -> tuple:
        """
        Split a video into spatial tiles as Vid objects. Optionally serialize tiles to disk with metadata.
        """
        arr = self.data.compute().values
        patches, metadata = patch(arr, z, overlap=overlap)

        dims = list(self.data.dims)
        tiles = []
        for patch_arr in patches:
            darr = da.from_array(patch_arr, chunks=patch_arr.shape)
            tile_da = xr.DataArray(darr, dims=dims)
            tiles.append(Vid.fromData(tile_da))

        metadata.update({
            "version": 1,
            "overlap": overlap,
            "dims": dims,
            "dtype": str(self.data.dtype),
            "tile_count": len(tiles),
            "tile_prefix": tile_prefix,
            "frame_prefix": frame_prefix,
            "suff": suff,
        })

        if outdir is not None:
            outdir = Path(outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            tile_entries = []
            for idx, tile in enumerate(tiles):
                tile_dir = outdir / f"{tile_prefix}{idx:04d}"
                if tile_dir.exists() and not overwrite:
                    raise FileExistsError(f"Directory {tile_dir} already exists and overwrite=False.")
                tile_dir.mkdir(parents=True, exist_ok=True)
                _write_vid_frames(tile, tile_dir, pre=frame_prefix, suff=suff, overwrite=overwrite)
                tile_entries.append({"index": idx, "path": tile_dir.name})
            metadata["tiles"] = tile_entries
            metadata_path = outdir / "metadata.json"
            if metadata_path.exists() and not overwrite:
                raise FileExistsError(f"File {metadata_path} already exists and overwrite=False.")
            metadata_path.write_text(json.dumps(metadata, indent=2))

        return tiles, metadata

    @classmethod
    def from_patches(cls, inpth: Path, chunks: Optional[tuple] = None) -> "Vid":
        """
        Reconstruct a Vid from a serialized patches directory containing tiles and metadata.json.
        """
        inpth = Path(inpth)
        metadata_path = inpth / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {inpth}")
        metadata = json.loads(metadata_path.read_text())
        if "tiles" in metadata:
            tile_paths = [inpth / t["path"] for t in metadata["tiles"]]
        else:
            tile_paths = sorted([p for p in inpth.iterdir() if p.is_dir()])
        suff = metadata.get("suff", ".jpg")
        patches_list = [Vid(p, suff=suff) for p in tile_paths]

        patch_arrays = [p.data.compute().values for p in patches_list]
        tiles = np.stack(patch_arrays, axis=0)
        hres, wres = metadata["patch_size"]
        stitched = unpatch(tiles, metadata, overlap=metadata.get("overlap"), hres=hres, wres=wres)
        target_dtype = np.dtype(metadata.get("dtype", "uint8"))
        if np.issubdtype(target_dtype, np.integer):
            stitched = np.clip(stitched, 0, 255)
        stitched = stitched.astype(target_dtype)

        if chunks is None:
            chunks = metadata.get("chunks", (1, 512, 512, -1))
        dims = metadata.get("dims", ["frame", "y", "x", "channel"])
        darr = da.from_array(stitched, chunks=chunks)
        data = xr.DataArray(darr, dims=dims)
        return cls.fromData(data)
