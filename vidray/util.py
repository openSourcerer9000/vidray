'yt-dlp -f "bv*+ba/b" --merge-output-format mkv {url}'
from funkshuns import cmder
from pathlib import Path

def vid2frames(inputVid: Path,outpth:Path, outputPattern='frames_%06d.jpg'):
    """
    Extract frames from a video file using ffmpeg.

    Args:
        inputVid (Path): Path to the input video file.
        outputPattern (str): Output filename pattern for extracted frames.
                                Example: 'frames_%06d.jpg' will create files like frames_000001.jpg, frames_000002.jpg, etc.
    """

    outpth.mkdir(exist_ok=True,parents= True)
    cmd = f'ffmpeg -i {inputVid} -vsync 0 {outpth}/{outputPattern}'
    if ' ' in str(outpth):
        cmd = f'ffmpeg -i "{inputVid}" -vsync 0 "{outpth}/{outputPattern}"'
        print('You need to run this manually since your path has spaces in it:', cmd)
        raise ValueError('Path has spaces, run manually', cmd)
    cmder(*cmd.split())

import os
import shutil
def savevid(frmpth,outvid,fps=30,suff='.jpg'):
    opth = frmpth.parent / 'temp_vid_frames'
    shutil.rmtree(opth,ignore_errors=True)
    opth.mkdir(exist_ok=True)
    os.chdir(str(frmpth.parent))
    frms = sorted(list(frmpth.glob('*'+suff)))

    fnms = [f'frame{str(i).zfill(6)}.png' for i in range(len(frms)) ]
    [shutil.copy(frm, opth/fn) for frm,fn in zip(frms,fnms)]
    Path(outvid).unlink(missing_ok=True)
    cmd = f'C:\\Py\\ffmpeg\\bin\\ffmpeg -framerate {fps} -i {opth.name}/frame%06d.png -c:v libx264 -pix_fmt yuv420p {outvid}'
    cmder(*cmd.split())
    shutil.rmtree(opth,ignore_errors=True)
    print(f'Saved video: {outvid}')