#!/usr/bin/env python3
"""
test_vidray.py

Pytest module for testing arithmetic operations (add, sub, mul, truediv)
between mock PNGs using the vidray module.
"""

import os
from pathlib import Path
import numpy as np
import imageio
import pytest
from vidray import Img, Vid
from dask.distributed import Client


# --- Constants for dummy image values ---
HEIGHT, WIDTH, CHANNELS = 10, 10, 3
VAL1 = 100  # constant value for first image
VAL2 = 50   # constant value for second image

# Expected results after clipping to [0, 255] (all operations are pixel-wise)
EXPECTED_RESULTS = {
    "add": 150,      # 100 + 50 = 150
    "sub": 50,       # |100 - 50| = 50
    "mul": 255,      # 100 * 50 = 5000 -> clip to 255
    "truediv": 2,    # 100 / 50 = 2.0 -> cast to 2
}


# --- Parametrized tests for Img arithmetic operations ---

@pytest.mark.parametrize("op, expected", [
    ("add", EXPECTED_RESULTS["add"]),
    ("sub", EXPECTED_RESULTS["sub"]),
    ("mul", EXPECTED_RESULTS["mul"]),
    ("truediv", EXPECTED_RESULTS["truediv"]),
])
def test_img_ops(imgPair, op, expected):
    img1, img2 = imgPair
    if op == "add":
        resImg = img1 + img2
    elif op == "sub":
        resImg = img1 - img2
    elif op == "mul":
        resImg = img1 * img2
    elif op == "truediv":
        resImg = img1 / img2
    else:
        pytest.fail("Unknown operator")
    npRes = resImg.compute().values
    assert np.all(npRes == expected), f"{op} op failed on Img"

# --- Parametrized tests for Vid arithmetic operations ---

@pytest.mark.parametrize("op, expected", [
    ("add", EXPECTED_RESULTS["add"]),
    ("sub", EXPECTED_RESULTS["sub"]),
    ("mul", EXPECTED_RESULTS["mul"]),
    ("truediv", EXPECTED_RESULTS["truediv"]),
])
def test_vid_ops(vidPair, op, expected, tmp_path):
    vid1, vid2 = vidPair
    if op == "add":
        resVid = vid1 + vid2
    elif op == "sub":
        resVid = vid1 - vid2
    elif op == "mul":
        resVid = vid1 * vid2
    elif op == "truediv":
        resVid = vid1 / vid2
    else:
        pytest.fail("Unknown operator")
    preview = resVid.frame(0)
    assert np.all(preview == expected), f"{op} op failed on Vid preview"
    
    outDir = tmp_path / "output"
    resVid.writePngs(outDir, pre="test_")
    written = list(outDir.glob("*.png"))
    assert len(written) == 3, "writePngs did not write expected number of files"

# --- Explicit test for Vid addition (vid1 + vid2) ---

def test_vid_addition(vidPair, tmp_path):
    vid1, vid2 = vidPair
    resVid = vid1 + vid2
    preview = resVid.frame(0)
    assert np.all(preview == EXPECTED_RESULTS["add"]), "Vid addition did not produce expected result"
    
    outDir = tmp_path / "vid_add_output"
    resVid.writePngs(outDir, pre="add_")
    written = list(outDir.glob("*.png"))
    assert len(written) == 3, "Incorrect number of output PNGs for vid addition"
