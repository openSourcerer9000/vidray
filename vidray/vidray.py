#!/usr/bin/env python3
"""
vidlib.py

A lazy video processing library using xarray and Dask distributed. This module
defines two main classes, Img and Vid, which represent a single image frame and
a collection of frames (video), respectively. Operations (addition, subtraction,
multiplication, division) are implemented via a generic binary operation helper
and remain lazy until a preview or write is triggered.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import xarray as xr
import dask.array as da
from dask import delayed
from dask.distributed import Client, progress
import imageio
import operator
from subprocess import Popen
from PIL import Image

def Cliente(n_workers: int = 6, threads_per_worker: int = 1,**kwargz) -> Client:
    """
    Launches a local Dask cluster with the specified number of workers and threads per worker.
    Prints the dashboard link and opens it in Windows Explorer.

    :param n_workers: Number of worker processes.
    :param threads_per_worker: Number of threads per worker process.
    :return: The launched Dask client.
    
    Example:
        >>> client = startDaskClient(n_workers=4, threads_per_worker=2)

    There's no single perfect magic number for every workload, but here are some general guidelines for a 6-core (12-thread) CPU:

    Start with one worker per physical core

    For example:

    python
    Copy
    client = Client(n_workers=6, threads_per_worker=1)
    This ensures each worker is pinned to one core. If your tasks are heavily CPU-bound (pure computation, little I/O), this often gives good performance without hyperthread contention.

    If tasks are not purely CPU-bound

    You can try increasing threads per worker (e.g., threads_per_worker=2) or adding more workers (e.g., 8 or 12). This can help if tasks spend a lot of time waiting on I/O or network operations.

    For instance:

    python
    Copy
    client = Client(n_workers=6, threads_per_worker=2)
    This might saturate the CPU better if there is some waiting in each task.

    Watch CPU usage, memory, and overall throughput

    If you see your CPU pinned at 100% for long stretches and tasks aren't finishing faster, you may be oversubscribing threads.

    If you see idle CPU time (e.g., your CPU usage is low), but tasks are slow, you might not have enough workers/threads to keep all cores busy.

    Try a few configurations

    Because every workload is different, it's often best to experiment with a few (n_workers, threads_per_worker) combinations (e.g., (6,1), (3,2), (6,2), (12,1)) and measure real-world performance (time to completion, CPU utilization, memory usage).

    Consider memory constraints

    More workers each require overhead memory for Python processes. If you have limited RAM, too many workers can lead to memory swapping or overhead from GC.

    Conversely, fewer workers with many threads can reduce process overhead, but can also lead to more Python GIL contention if tasks are pure Python loops.

    Practical Starting Point

    CPU-bound numeric workloads: n_workers=6, threads_per_worker=1

    I/O or mixed workloads: n_workers=6, threads_per_worker=2

    Check performance, CPU usage, and memory usage in each scenario. Whichever setting completes your real tasks fastest is the 'best' configuration for your environment.
    """
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker,**kwargz)
    print(f"Dask client dashboard at: {client.dashboard_link}")
    # Open the dashboard in Windows Explorer.
    Popen(f'explorer "{client.dashboard_link}"')
    return client

def _parallelBinaryOp(xr1: xr.DataArray,
                      xr2: xr.DataArray,
                      op,
                      absResult: bool = False,
                      intermediate=np.int16) -> xr.DataArray:
    """
    Generic helper to apply a binary operator in parallel using xarray + Dask.
    """
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
        # If 'other' is a scalar, wrap it in an xarray.DataArray of the same shape
        # or rely on xarray broadcasting with an empty shape
        if isinstance(other, (int, float)):
            otherData = xr.DataArray(other)  # shape=()
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

class Img(BinaryOperable):
    """
    Single-frame image class backed by a lazy xarray.DataArray.
    
    :param pth: Optional Path to a PNG file. If provided, the image is loaded lazily.
    :param dataArr: Optional pre-existing xarray.DataArray.
    :param chunks: Chunk sizes for the frame (e.g. (512, 512, -1)).
    
    Example::
        >>> im = Img(pth=Path("example.png"))
    """
    def __init__(self, pth: Optional[Path] = None, dataArr: Optional[xr.DataArray] = None, chunks: tuple = (512, 512, -1)):
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
        
        :param dataArr: DataArray to wrap.
        :return: Img instance.
        """
        obj = cls.__new__(cls)
        obj.data = dataArr
        return obj

    def compute(self) -> xr.DataArray:
        """
        Computes the underlying DataArray.
        
        :return: Computed DataArray.
        """
        return self.data.compute()

    def toNumpy(self) -> np.ndarray:
        """
        Returns the computed image as a NumPy array.
        
        :return: NumPy array.
        """
        return self.compute().values

    def __add__(self, other: "Img") -> "Img":
        """Adds two images pixel-wise."""
        return self._applyBinaryOp(other, operator.add)

    def __sub__(self, other: "Img") -> "Img":
        """Subtracts two images and returns the absolute difference."""
        return self._applyBinaryOp(other, operator.sub, absResult=True)

    def __mul__(self, other: "Img") -> "Img":
        """Multiplies two images pixel-wise."""
        return self._applyBinaryOp(other, operator.mul, 
                                #    intermediate=np.int32
                                   )

    def __truediv__(self, other: "Img") -> "Img":
        """Divides two images pixel-wise."""
        return self._applyBinaryOp(other, operator.truediv, intermediate=np.float32)

class Vid(BinaryOperable):
    """
    Video-like collection of frames stored as a lazy xarray.DataArray with dims
    ["frame", "y", "x", "channel"]. Operations are lazy and computed only on preview or write.
    
    :param pathsOrImgs: List of Paths (for lazy reading) or Img objects.
    :param chunks: Chunk sizes for the video array (e.g. (1, 512, 512, -1)).
    
    Example::
        >>> vid = Vid(list(Path("frames").glob("*.png")))
    """
    def __init__(self, pathsOrImgs: List[Union[Path, Img]], chunks: tuple = (1, 512, 512, -1)):
        if not pathsOrImgs:
            raise ValueError("No frames provided to Vid.")
        if isinstance(pathsOrImgs, Path):
            if pathsOrImgs.is_dir():
                pathsOrImgs = list(pathsOrImgs.glob("*.png"))
                assert len(pathsOrImgs) > 0, "No PNG files found in directory. MAybe another extension?"
            else:
                pathsOrImgs = [pathsOrImgs]
        if isinstance(pathsOrImgs[0], Path):
            example = imageio.imread(str(pathsOrImgs[0]))
            frameShape = example.shape
            frameDtype = example.dtype
            daList = [
                da.from_delayed(delayed(imageio.imread)(str(p)), shape=frameShape, dtype=frameDtype)
                for p in pathsOrImgs
            ]
            stacked = da.stack(daList, axis=0)
            self.data = xr.DataArray(stacked, dims=["frame", "y", "x", "channel"]).chunk(chunks)
        elif isinstance(pathsOrImgs[0], Img):
            dataList = [img.data.expand_dims(dim={"frame": [i]}) for i, img in enumerate(pathsOrImgs)]
            self.data = xr.concat(dataList, dim="frame").chunk(chunks)
        else:
            raise TypeError("pathsOrImgs must be a list of pathlib.Path or Img objects.")

    @classmethod
    def fromData(cls, dataArr: xr.DataArray) -> "Vid":
        """
        Creates a Vid instance from an existing DataArray.
        
        :param dataArr: DataArray with dims ["frame", "y", "x", "channel"].
        :return: Vid instance.
        """
        vidObj = cls.__new__(cls)
        vidObj.data = dataArr
        return vidObj

    def frame(self, frame: int = 0) -> np.ndarray:
        """
        Computes and returns a single frame as a NumPy array.
        
        :param frame: Frame index.
        :return: PIL Image for the selected frame.
        """
        return Image.fromarray( self.data.isel(frame=frame).compute().values)
    
    def writePngs(self, outDest: Union[Path, list], pre: str = "frame", overwrite: bool = False) -> None:
        """
        Writes all frames as PNG images in parallel.

        If outDest is a Path (directory), filenames are generated as:
        pre + {frame:08d}.png
        within that directory.
        If outDest is a list of Paths, these are used directly.

        :param outDest: Either a directory (Path) to write files into or a list of file Paths.
        :param pre: Filename prefix (used only when outDest is a directory).
        :param overwrite: If False, raises an error if a target file already exists.
        """
        from dask import delayed
        # Determine target file paths
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
        
        def writeFrame(npFrame: np.ndarray, outPth: Path, overwrite: bool) -> str:
            if outPth.exists() and not overwrite:
                raise FileExistsError(f"File {outPth} already exists and overwrite is False.")
            imageio.imwrite(str(outPth), npFrame)
            return f"Saved {outPth.name}"
        
        tasks = []
        for i in range(nFrames):
            frameDA = self.data.isel(frame=i)
            frameDelayed = delayed(frameDA.compute)()
            tasks.append(delayed(writeFrame)(frameDelayed, filePaths[i], overwrite))
        
        futures = Client.current().compute(tasks)
        print("Writing PNGs in parallel; check Dask dashboard for progress.")
        progress(futures)
        results = Client.current().gather(futures)
        return results
        # for r in results:
        #     print(r)


    def visualizeGraph(self, filename: str = "vid_graph.png") -> None:
        """
        Visualizes the underlying Dask task graph and saves it to a file.
        
        :param filename: Output filename for the graph.
        """
        self.data.data.visualize(filename=filename)
        print(f"Dask graph saved to {filename}")

    def compute(self) -> xr.DataArray:
        """
        Computes the entire video dataset.
        
        :return: Computed xarray.DataArray.
        """
        return self.data.compute()

    def __add__(self, other: "Vid") -> "Vid":
        """Adds two videos frame-by-frame."""
        return self._applyBinaryOp(other, operator.add)

    def __sub__(self, other: "Vid") -> "Vid":
        """Subtracts two videos frame-by-frame, returning the absolute difference."""
        return self._applyBinaryOp(other, operator.sub, absResult=True)

    def __mul__(self, other: "Vid") -> "Vid":
        """Multiplies two videos frame-by-frame."""
        return self._applyBinaryOp(other, operator.mul, 
                                #    intermediate=np.int32
                                   )

    def __truediv__(self, other: "Vid") -> "Vid":
        """Divides two videos frame-by-frame."""
        return self._applyBinaryOp(other, operator.truediv, intermediate=np.float32)
