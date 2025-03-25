#!/usr/bin/env python3
"""
test_transforms.py

Expanded Pytest module for verifying the 'scale' operation in the Scalable mixin
(used by Vid and Img classes). We specifically test 4-channel data (e.g. RGBA),
and verify pixel values inside/outside the scaled region.
"""

import pytest
import numpy as np
import xarray as xr

# Import the constants and classes (update to your actual import paths)
# from transforms import RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS
# from vidlib import Vid, Img
from vidray import Vid, Img, RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS

#############################################
# Fixtures
#############################################

@pytest.fixture
def sampleImg4Chan():
    """
    Returns an Img with shape (y=10, x=10, channel=4), containing RGBA = (10, 20, 30, 255).
    """
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    # Fill with a distinct RGBA color
    arr[..., 0] = 10   # R
    arr[..., 1] = 20   # G
    arr[..., 2] = 30   # B
    arr[..., 3] = 255  # A
    dataArr = xr.DataArray(arr, dims=["y", "x", "channel"])
    return Img.fromData(dataArr)

@pytest.fixture
def sampleVid4Chan():
    """
    Returns a Vid with shape (frame=2, y=10, x=10, channel=4), each frame = (10,20,30,255).
    """
    arr = np.zeros((2, 10, 10, 4), dtype=np.uint8)
    arr[..., 0] = 10   # R
    arr[..., 1] = 20   # G
    arr[..., 2] = 30   # B
    arr[..., 3] = 255  # A
    dataArr = xr.DataArray(arr, dims=["frame", "y", "x", "channel"])
    return Vid.fromData(dataArr)

#############################################
# Helpers
#############################################

def getShape(obj):
    """
    For Vid: returns (frames, height, width, channels).
    For Img: returns (height, width, channels).
    """
    if isinstance(obj, Vid):
        return (
            obj.data.sizes["frame"],
            obj.data.sizes["y"],
            obj.data.sizes["x"],
            obj.data.sizes["channel"]
        )
    elif isinstance(obj, Img):
        return (
            obj.data.sizes["y"],
            obj.data.sizes["x"],
            obj.data.sizes["channel"]
        )
    else:
        raise TypeError("Unknown object type")

def assertRegionColor(array, yStart, yEnd, xStart, xEnd, expectedRGBA):
    """
    Ensures that array[yStart:yEnd, xStart:xEnd, :] is all `expectedRGBA`.
    """
    region = array[yStart:yEnd, xStart:xEnd, :]
    # region has shape (height, width, channels)
    # Create a reference array of the same shape, filled with expectedRGBA
    ref = np.full(region.shape, expectedRGBA, dtype=region.dtype)
    # Now compare
    np.testing.assert_array_equal(region, ref)


def assertRegionZero(array, yStart, yEnd, xStart, xEnd):
    """
    Checks that array[yStart:yEnd, xStart:xEnd, :] is all zeros.
    """
    region = array[yStart:yEnd, xStart:xEnd, :]
    np.testing.assert_array_equal(region, 0)

#############################################
# Parametrized Tests for 4-Channel Data
#############################################

@pytest.mark.parametrize("mode", [RESIZE_IMAGE, RESIZE_CANVAS])
@pytest.mark.parametrize("scaleFactor", [0.5, 2.0])
def test_scale_img4chan_pixels(sampleImg4Chan, mode, scaleFactor):
    """
    Demonstrates checking actual pixel values when scaling a 4-channel Img
    in either RESIZE_IMAGE or RESIZE_CANVAS mode, with scaleFactor=0.5 or 2.0.
    We'll assume 'nearest' interpolation so scaled region keeps the same color.
    """
    # Original shape => (10,10,4)
    # RGBA = (10,20,30,255)
    out = sampleImg4Chan.scale(scaleFactor=scaleFactor, mode=mode, interpolation="nearest")

    outArr = out.compute().values  # shape depends on mode & scale factor
    # We'll handle each scenario separately:

    if mode == RESIZE_IMAGE:
        # Canvas stays 10x10, the scaled image is pasted at top-left (0,0).
        # If scaleFactor=0.5 => scaled region is 5x5
        # If scaleFactor=2.0 => scaled region is 20x20, but clipped to 10x10

        if scaleFactor == 0.5:
            # The top-left 5x5 should remain RGBA=(10,20,30,255)
            assertRegionColor(outArr, 0, 5, 0, 5, [10,20,30,255])
            # The rest => zero
            assertRegionZero(outArr, 5, 10, 0, 10)  # below the scaled region
            assertRegionZero(outArr, 0, 10, 5, 10)  # to the right
        else:
            # scaleFactor=2.0 => 20x20 scaled region, but clipped to 10x10
            # entire 10x10 canvas should be the original color
            assertRegionColor(outArr, 0, 10, 0, 10, [10,20,30,255])

    elif mode == RESIZE_CANVAS:
        # The original image remains 10x10, placed on a bigger or smaller canvas
        # at pivot top-left (0,0).
        # If scaleFactor=0.5 => new canvas => (5,5)? Actually "scaleFactor" means
        # the canvas is scaled, but we do the same logic as _scaleCanvasOnly.
        # We'll see how your code calculates it: newH=5, newW=5 => final shape => (5,5)
        # The original 10x10 is partially clipped if scaleFactor<1. 
        # If scaleFactor>1 => bigger canvas.

        outShape = getShape(out)
        if scaleFactor == 0.5:
            # The new canvas is (5,5).
            # The original content is 10x10 => only top-left 5x5 is valid => partial clip
            # outArr => shape (5,5,4)
            assert outShape == (5,5,4)
            # We expect the top-left portion to match the original's top-left 5x5 region
            # The rest doesn't exist in the new array, so no extra checks needed
            # We can check all is the original color, or we can check partial.
            # Because your code might paste only the first 5x5 from the original
            assertRegionColor(outArr, 0, 5, 0, 5, [10,20,30,255])
        else:
            # scaleFactor=2.0 => new canvas => (20,20)
            # The original 10x10 is placed at (0,0). The rest is zero
            assert outShape == (20,20,4)
            # top-left 10x10 => original color
            assertRegionColor(outArr, 0, 10, 0, 10, [10,20,30,255])
            # the rest => zero
            assertRegionZero(outArr, 10, 20, 0, 20)
            assertRegionZero(outArr, 0, 20, 10, 20)



# class DummyImg:
#     def __init__(self, data: xr.DataArray):
#         self.data = data
#     @classmethod
#     def fromData(cls, data: xr.DataArray) -> "DummyImg":
#         return cls(data)
#     def compute(self):
#         return self.data.compute() if hasattr(self.data, "compute") else self.data

@pytest.fixture
def sampleImg4Chan():
    arr = np.full((10,10,4), [10,20,30,255], dtype=np.uint8)
    da = xr.DataArray(arr, dims=["y","x","channel"])
    return Img.fromData(da)



# The following TEST_CASES list is one possible set.
# These expected regions are computed under one interpretation of:
#   anchor = canvasPivot - scaledPivot,
# where for each axis:
#   if key "top" is provided, pivot = value; if "bottom" is provided, pivot = size - value.
# (Adjust the numbers as needed for your actual pivot logic.)

#!/usr/bin/env python3
"""
test_transformz4.py

Tests for the pivot behavior of 4‑channel scaling.
Assume:
  - A DummyImg object with .data as a 10×10, 4‑channel xarray.DataArray
    whose pixels all equal [10,20,30,255].
  - scaleFactor=0.5 yields a 5×5 scaled image.
  - The pivot dict is interpreted as the pivot point in both the scaled image
    and canvas. The paste anchor = canvasPivot – scaledPivot.
    
For example:
    - pivot {"top":0, "left":0} yields anchor (0,0) so that rows 0–5,cols 0–5 are colored.
    - pivot {"bottom":"2px","right":"2px"} is interpreted as “off‐canvas” (no overlap)
      and so the entire canvas is zero.
"""


# The following TEST_CASES list is one possible set.
# These expected regions are computed under one interpretation of:
#   anchor = canvasPivot - scaledPivot,
# where for each axis:
#   if key "top" is provided, pivot = value; if "bottom" is provided, pivot = size - value.
# (Adjust the numbers as needed for your actual pivot logic.)

TEST_CASES = [
    # pivot, zeroPts, colorPts
    ({"top": 0, "left": 0},
        [(5,10,0,10), (0,5,5,10)],
        [(0,5,0,5)]),
    ({"bottom": "2px", "right": "2px"},
        [(0,10,0,10)],
        []),
    ({"bottom": 0, "left": "0px"},
        [(0,5,0,10), (5,10,5,10)],
        [(5,10,0,5)]),
    ({"top": "20%", "right": 0},
        [(0,1,0,10), (6,10,0,10), (1,6,0,5)],
        [(1,6,5,10)]),
    ({"bottom": "40%", "left": "20%"},
        [(0,3,0,10), (8,10,0,10), (3,8,0,1), (3,8,6,10)],
        [(3,8,1,6)]),
    ({"top": "10px", "left": "5%"},
        [(0,10,0,10)],
        []),
    ({"bottom": "15%", "right": "10px"},
        [(0,10,0,10)],
        []),
    ({"top": 0, "right": "25%"},
        [(0,10,0,4)],
        [(0,5,4,9)]),
    ({"bottom": "5px", "left": 0},
        [(0,5,0,10), (5,10,5,10)],
        [(5,10,0,5)]),
    ({"top": "50%", "left": "50%"},
        [(0,2,0,10), (7,10,0,10), (2,7,0,2), (2,7,7,10)],
        [(2,7,2,7)]),
    ({"bottom": 0, "right": "0px"},
        [(0,5,0,10), (5,10,0,5)],
        [(5,10,5,10)]),
    ({"top": "30%", "left": "10px"},
        [(0,10,0,10)],
        []),
    ({"bottom": "60%", "right": "15%"},
        [(0,10,0,10)],
        [(2,7,4,9)]),  # computed anchor = (2,4) → pasted block covers rows 2–7, cols 4–9
    ({"top": "5%", "right": 0},
        [(0,1,0,10), (6,10,0,10), (1,6,0,5)],
        [(1,6,5,10)]),
    ({"bottom": "25px", "left": "30%"},
        [(0,10,0,10)],
        [(5,10,1,6)]),
]

@pytest.mark.parametrize("pivot,zeroPts,colorPts", TEST_CASES)
def testScaleImg4ChanPivot(sampleImg4Chan, pivot, zeroPts, colorPts):
    outImg = sampleImg4Chan.scale(scaleFactor=0.5, mode=RESIZE_IMAGE, pvt=pivot, interpolation="nearest")
    outArr = outImg.compute().values  # expected shape (10,10,4)
    for (y1, y2, x1, x2) in zeroPts:
        assertRegionZero(outArr, y1, y2, x1, x2)
    for (y1, y2, x1, x2) in colorPts:
        assertRegionColor(outArr, y1, y2, x1, x2, [10,20,30,255])
