import pytest
import vidray as vr
from pathlib import Path
import pandas as pd, numpy as np

import rootpath
root = Path(rootpath.detect())
mc = root /  'tests' / 'mock'

@pytest.mark.parametrize("z", 
                         [2, 3,(2,3), 4, (3,2)]
                         )
def test_patch_shapes(z):
    vpth = mc / 'ants' 
    fpth = vpth / 'frames'

    # client = vr.Cliente()
    vid= vr.Vid(fpth)
    vid.patch(z,outdir=fpth.parent/'pch')

    pvid = vr.Vid.from_patches(fpth.parent/'pch')
    for f in range(len(vid)):
        a = vid.frame(f)
        b = pvid.frame(f)
        diff = np.array(a) - np.array(b)
        same = np.median(diff) < 8
        if not same:
            print(f"Frame {f} differs",np.median(diff))
            a.show()
            b.show()
            assert same