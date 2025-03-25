#!/usr/bin/env python3
"""
test_transforms.py

Pytest module for verifying the 'scale' operation in the Scalable mixin
(used by Vid and Img classes). We test different modes (RESIZE_BOTH,
RESIZE_IMAGE, RESIZE_CANVAS), pivot points, and dimension overrides.
"""

import pytest
import numpy as np
import xarray as xr

# Import the constants and classes
# from transforms import 
from vidray import Vid, Img, RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS

#############################
# Fixtures
#############################

@pytest.fixture
def sampleVid():
    """
    Returns a Vid with shape (frame=2, y=10, x=10, channel=3), filled with 128.
    """
    arr = np.full((2, 10, 10, 3), 128, dtype=np.uint8)
    dataArr = xr.DataArray(arr, dims=["frame", "y", "x", "channel"])
    return Vid.fromData(dataArr)

@pytest.fixture
def sampleImg():
    """
    Returns an Img with shape (y=10, x=10, channel=3), filled with 128.
    """
    arr = np.full((10, 10, 3), 128, dtype=np.uint8)
    dataArr = xr.DataArray(arr, dims=["y", "x", "channel"])
    return Img.fromData(dataArr)

#############################
# Helper: get shape
#############################

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

#############################
# Tests for Vid scaling
#############################

@pytest.mark.parametrize("mode", [RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS])
def test_scaleVid_factor(sampleVid, mode):
    """
    Scale the Vid by factor=2.0, check shape changes according to mode.
    Original shape => (2, 10, 10, 3).
    """
    out = sampleVid.scale(scaleFactor=2.0, mode=mode)
    if mode == RESIZE_BOTH:
        # Entire frame from (10,10) to (20,20)
        assert getShape(out) == (2, 20, 20, 3)
    elif mode == RESIZE_IMAGE:
        # Canvas remains 10x10, scaled image is "pasted" within it
        assert getShape(out) == (2, 10, 10, 3)
    elif mode == RESIZE_CANVAS:
        # Canvas becomes (20,20), original image remains 10x10 inside
        assert getShape(out) == (2, 20, 20, 3)

@pytest.mark.parametrize("mode", [RESIZE_BOTH, RESIZE_IMAGE, RESIZE_CANVAS])
def test_scaleVid_width(sampleVid, mode):
    """
    Scale the Vid by specifying width=15 only (height is auto-derived from aspect).
    Original aspect is 1.0 => new => (15,15).
    """
    out = sampleVid.scale(width=15, mode=mode)
    if mode == RESIZE_BOTH:
        assert getShape(out) == (2, 15, 15, 3)
    elif mode == RESIZE_IMAGE:
        # Canvas remains (10,10)
        assert getShape(out) == (2, 10, 10, 3)
    elif mode == RESIZE_CANVAS:
        assert getShape(out) == (2, 15, 15, 3)

@pytest.mark.parametrize("pivot", [
    {"top": 0, "left": 0},
    {"bottom": 0, "right": "0%"},
    {"bottom": "10px", "right": "20%"}
])
def test_scaleVid_pivot(sampleVid, pivot):
    """
    Pivot mainly affects how scaled content or canvas is anchored, but we
    can at least confirm shape remains correct in a given mode.
    """
    # We'll test pivot with RESIZE_IMAGE for demonstration
    out = sampleVid.scale(scaleFactor=1.5, mode=RESIZE_IMAGE, pvt=pivot)
    # Canvas remains (10,10), shape => (2,10,10,3)
    assert getShape(out) == (2, 10, 10, 3)

def test_scaleVid_noParams(sampleVid):
    """
    Expect ValueError if neither scaleFactor nor width/height is provided.
    """
    with pytest.raises(ValueError, match="Must provide scaleFactor or width/height"):
        sampleVid.scale()

#############################
# Tests for Img scaling
#############################

def test_scaleImg_factor(sampleImg):
    """
    Scale the Img by factor=2.0 (RESIZE_BOTH).
    Original shape => (10,10,3) => new => (20,20,3).
    """
    out = sampleImg.scale(scaleFactor=2.0)
    assert getShape(out) == (20, 20, 3)

def test_scaleImg_width(sampleImg):
    """
    Scale the Img by specifying width=5 => new => (5,5,3).
    """
    out = sampleImg.scale(width=5)
    assert getShape(out) == (5, 5, 3)

def test_scaleImg_bothDims(sampleImg):
    """
    Scale an Img by specifying both width=12, height=8 => no aspect lock.
    """
    out = sampleImg.scale(width=12, height=8)
    assert getShape(out) == (8, 12, 3)
