'yt-dlp -f "bv*+ba/b" --merge-output-format mkv {url}'
from funkshuns import cmder
from pathlib import Path

# def vid2frames(inputVid: Path,outpth:Path, outputPattern='frames_%06d.jpg'):
#     """
#     Extract frames from a video file using ffmpeg.

#     Args:
#         inputVid (Path): Path to the input video file.
#         outputPattern (str): Output filename pattern for extracted frames.
#                                 Example: 'frames_%06d.jpg' will create files like frames_000001.jpg, frames_000002.jpg, etc.
#     """

#     outpth.mkdir(exist_ok=True,parents= True)
#     cmd = f'ffmpeg -i {inputVid} -vsync 0 {outpth}/{outputPattern}'
#     if ' ' in str(outpth):
#         cmd = f'ffmpeg -i "{inputVid}" -vsync 0 "{outpth}/{outputPattern}"'
#         print('You need to run this manually since your path has spaces in it:', cmd)
#         raise ValueError('Path has spaces, run manually', cmd)
#     cmder(*cmd.split())
from fractions import Fraction



def video_fps(pth: Path | str) -> float:
    rate = subprocess.check_output(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "default=nw=1:nk=1",
            str(pth),
        ],
        text=True,
    ).strip()

    return float(Fraction(rate))
def vid2frames(
    inputVid: Path | str,
    outpth: Path | str,
    outputPattern: str = "frames_%06d.jpg",
    x: int = 3,
    base_fps: int = 30,
    max_w: int = 1280,
    max_h: int = 720,
    quality: int = 2,
):
    """
    Extract every x frames at base_fps, scaling only upward for higher FPS.

    x=3, base_fps=30:
        24 fps -> every 3 frames
        30 fps -> every 3 frames
        60 fps -> every 6 frames
    """
    inputVid, outpth = Path(inputVid), Path(outpth)
    outpth.mkdir(exist_ok=True, parents=True)

    fps = video_fps(inputVid)
    every = max(x, round(x * fps / base_fps))

    vf = (
        f"select='not(mod(n,{every}))',"
        f"scale='min({max_w},iw)':'min({max_h},ih)':"
        "force_original_aspect_ratio=decrease"
    )

    cmder(
        "ffmpeg",
        "-i", str(inputVid),
        "-vf", vf,
        "-fps_mode", "vfr",
        "-q:v", str(quality),
        str(outpth / outputPattern),
    )

    return every

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

import subprocess
from pathlib import Path

def ytdl(source, target_format='a',outpth='cwd', extra_args=None):
    """
    ... docstring ...
    """
    # Path source for better checking
    src_path = Path(source)

    # Determine URLs to download
    if src_path.is_file():
        urls_to_process = [line.strip() for line in src_path.read_text().splitlines()]
    else:
        urls_to_process = [source]

    # Determine base flags for format
    if target_format == 'v':
        format_flags = ['-f', 'bv+ba/b']
    else: # Default to audio
        format_flags = ['-x']

    base_command = ['yt-dlp'] + format_flags
    
    # Process extra_args
    if extra_args:
        for flag, value in extra_args.items():
            # Handle flags like {'-x': True} or {'--embed-metadata': None}
            if value is True or isinstance(value, type(None)):
                base_command.append(flag)
            else:
                 # Handle key-value pairs
                base_command.extend([flag, str(value)])

    # Download loop
    for url in urls_to_process:
        if not url: continue # Skip empty lines
        print(f"Downloading: {url}")
        
        try:
            command_to_run = base_command + [url]
            subprocess.run(command_to_run, check=True) # check=True raises CalledProcessError on failure
        except FileNotFoundError:
            print("Error: yt-dlp command not found. Please install it with 'pip install yt-dlp'.")
            break
        except subprocess.CalledProcessError as e:
            print(f"yt-dlp failed for URL '{url}' with return code {e.returncode}.")
            print("--- Standard Error ---\n" + e.stderr.decode() if e.stderr else "")
        except Exception as e:
            print(f"An unexpected error occurred for URL '{url}': {e}")

