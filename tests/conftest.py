# tests/conftest.py
import os
import numpy as np
import pytest
import imageio

from pathlib import Path
from dask.distributed import Client

from vidray import Img, Vid  # or wherever your classes are imported from

# --- Constants for dummy image values ---
HEIGHT, WIDTH, CHANNELS = 10, 10, 3
VAL1 = 100  # constant value for first image
VAL2 = 50   # constant value for second image

@pytest.fixture(scope="session", autouse=True)
def daskClient():
    """
    Starts a local Dask distributed client for the test session.
    """
    client = Client(processes=False, threads_per_worker=1, n_workers=1)
    yield client
    client.close()

@pytest.fixture(scope="session")
def mockDirs():
    """
    Creates a 'mock' directory containing two subdirectories,
    each with dummy PNG images. Returns (dir1, dir2).
    """
    # The directory that holds this conftest.py
    mc = Path(__file__).parent / "mock"
    mc.mkdir(parents=True, exist_ok=True)

    dir1 = mc / "dir1"
    dir2 = mc / "dir2"
    dir1.mkdir(exist_ok=True)
    dir2.mkdir(exist_ok=True)
    numFrames = 3

    # Create dummy PNGs in both directories.
    for i in range(numFrames):
        arr1 = np.full((HEIGHT, WIDTH, CHANNELS), VAL1, dtype=np.uint8)
        arr2 = np.full((HEIGHT, WIDTH, CHANNELS), VAL2, dtype=np.uint8)
        imageio.imwrite(str(dir1 / f"frame_{i:02d}.png"), arr1)
        imageio.imwrite(str(dir2 / f"frame_{i:02d}.png"), arr2)
    return dir1, dir2

@pytest.fixture
def vidPair(mockDirs):
    """
    Returns a pair of Vid objects created from the two mock directories.
    """
    dir1, dir2 = mockDirs
    vid1 = Vid(sorted(dir1.glob("*.png")))
    vid2 = Vid(sorted(dir2.glob("*.png")))
    return vid1, vid2

@pytest.fixture
def imgPair(tmp_path):
    """
    Creates two temporary PNG files with constant values and returns a pair of Img objects.
    """
    file1 = tmp_path / "img1.png"
    file2 = tmp_path / "img2.png"
    arr1 = np.full((HEIGHT, WIDTH, CHANNELS), VAL1, dtype=np.uint8)
    arr2 = np.full((HEIGHT, WIDTH, CHANNELS), VAL2, dtype=np.uint8)
    imageio.imwrite(str(file1), arr1)
    imageio.imwrite(str(file2), arr2)
    return Img(pth=file1), Img(pth=file2)
