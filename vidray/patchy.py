import numpy as np
import math
from patchify import patchify  # Ensure you have patchify installed

def patch(imgOrVid, z, overlap=20):
    """
    Split an image or video into patches.

    Parameters:
      imgOrVid : array-like
          Input image (H,W,C) or video (F,H,W,C)
      z : int or (int, int)
          If int, both vertical and horizontal patch counts will be z.
          If tuple, interpreted as (zu, zv) for vertical and horizontal splits.
      overlap : int
          Number of pixels to overlap (and later blend) between adjacent patches.

    Returns:
      patches : np.ndarray
          Array of patches.
      metadata : dict
          Contains:
             - 'padded_shape': shape after padding,
             - 'original_shape': shape of input,
             - 'tiles_shape': raw shape from patchify,
             - 'step': step sizes used,
             - 'patch_size': (patch_h, patch_w),
             - 'grid': (zu, zv) as provided.
    """
    # Interpret z as (zu, zv): zu = vertical (rows), zv = horizontal (columns)
    if isinstance(z, int):
        zu, zv = z, z
    else:
        zu, zv = z

    a = np.array(imgOrVid)
    ogshp = a.shape  # original shape before padding
    # Determine if input is video (has frame dimension)
    f = ogshp[0] if len(ogshp) > 3 else None

    # Get spatial dimensions from the original input (before padding)
    h, w, c = ogshp[-3:]
    
    # Compute patch size and step such that final stitched dimensions equal the original.
    # (h - overlap) is divided evenly among the vertical patches, then add back the overlap.
    patch_h = math.ceil((h - overlap) / zu) + overlap
    patch_w = math.ceil((w - overlap) / zv) + overlap
    step_h = patch_h - overlap  # This equals ceil((h - overlap) / zu)
    step_w = patch_w - overlap  # This equals ceil((w - overlap) / zv)
    final_h = zu * step_h + overlap
    final_w = zv * step_w + overlap

    # Pad spatial dimensions (but not frames if video) to fit the final grid.
    if overlap:
        extra_h = max(0, final_h - (h + overlap))
        extra_w = max(0, final_w - (w + overlap))
        pad_shp = [(0, overlap + extra_h), (0, overlap + extra_w), (0, 0)]
        if f is not None:
            pad_shp = [(0, 0)] + pad_shp
        a = np.pad(a, pad_shp, mode='reflect')
    padded_shape = a.shape

    # Set winshape and step for patchify
    if f is not None:
        winshape = [ogshp[0], patch_h, patch_w, c]
        step = (ogshp[0], step_h, step_w, c)
    else:
        winshape = [patch_h, patch_w, c]
        step = (step_h, step_w, c)

    # Create patches with patchify
    tiles = patchify(a, winshape, step=step)
    tileshp = tiles.shape

    # Set grid explicitly from (zu, zv)
    grid = (zu, zv)

    # Reshape to a list of patches
    patches = tiles.reshape(-1, *winshape)
    
    # Save metadata for unpatching
    metadata = {
        'padded_shape': padded_shape,
        'original_shape': ogshp,
        'tiles_shape': tileshp,
        'step': step,
        'patch_size': (patch_h, patch_w),
        'grid': grid,
        'overlap': overlap
    }
    
    return patches, metadata

def create_symmetric_2dmask(h, w, overlap):
    """
    Creates a 2D mask of shape (h, w) that is 1.0 in the interior
    and fades linearly to 0.0 over 'overlap' pixels at each edge.
    """
    mask = np.ones((h, w), dtype=np.float32)
    if overlap <= 0:
        return mask
    # Fade top edge
    for row in range(overlap):
        alpha = row / overlap
        mask[row, :] *= alpha
    # Fade bottom edge
    for row in range(h - overlap, h):
        alpha = (h - row) / overlap
        mask[row, :] *= alpha
    # Fade left edge
    for col in range(overlap):
        alpha = col / overlap
        mask[:, col] *= alpha
    # Fade right edge
    for col in range(w - overlap, w):
        alpha = (w - col) / overlap
        mask[:, col] *= alpha
    return mask

def unpatch(patches, metadata, overlap=None, hres=None, wres=None):
    """
    Reconstruct a full image or video from a grid of patches.

    Uses the grid stored in metadata["grid"]. If hres/wres are not provided,
    they are taken from metadata["patch_size"].

    Parameters:
      patches : np.ndarray
          For a video: shape (num_tiles, f, patch_h, patch_w, C);
          For an image: shape (num_tiles, patch_h, patch_w, C).
      metadata : dict
          Must include:
             - "original_shape": original shape of the image/video.
             - "patch_size": a tuple (patch_h, patch_w) computed in patch().
             - "grid": the intended grid (zu, zv) as specified when patching.
      overlap : int
          The pixel overlap used during patching.
      hres : int, optional
          Height of each patch for reconstruction; if None, metadata["patch_size"][0] is used.
      wres : int, optional
          Width of each patch for reconstruction; if None, metadata["patch_size"][1] is used.

    Returns:
      np.ndarray : The fully stitched image (H, W, C) or video (f, H, W, C).

    The final stitched dimensions are computed as:
       final_H = zu * step_h + overlap
       final_W = zv * step_w + overlap
    """
    if overlap is None:
        overlap = metadata.get("overlap", 20)

    # Use patch size from metadata if not provided
    if hres is None or wres is None:
        hres, wres = metadata["patch_size"]
    
    # Use the explicit grid from metadata (this is (zu, zv))
    if "grid" not in metadata:
        raise ValueError("Metadata must contain a 'grid' key with the intended grid dimensions.")
    zu, zv = metadata["grid"]
    
    # For clarity, step_h and step_w can be computed as:
    step_h = hres - overlap
    step_w = wres - overlap

    # Debug prints so you can verify
    # print("DEBUG: Grid dimensions =", (zu, zv))
    # print("DEBUG: Patch size =", (hres, wres))
    
    ogshp = metadata["original_shape"]
    
    # Compute final stitched dimensions so that:
    # final_H = zu * step_h + overlap  and  final_W = zv * step_w + overlap
    final_H = zu * step_h + overlap
    final_W = zv * step_w + overlap

    # Check if we're working with video (4D original shape) or image (3D)
    if len(ogshp) == 4:
        f, H, W, C = ogshp
        out = np.zeros((f, final_H, final_W, C), dtype=np.float32)
        weight = np.zeros((f, final_H, final_W, C), dtype=np.float32)
        mask = create_symmetric_2dmask(hres, wres, overlap)[..., None]
        idx = 0
        for i in range(zu):
            for j in range(zv):
                top = i * step_h
                left = j * step_w
                # Each patch is assumed to be (f, hres, wres, C)
                out[:, top:top+hres, left:left+wres, :] += patches[idx] * mask
                weight[:, top:top+hres, left:left+wres, :] += mask
                idx += 1
        out /= np.maximum(weight, 1e-8)
        return out[:, :H, :W, :]
    else:
        H, W, C = ogshp
        out = np.zeros((final_H, final_W, C), dtype=np.float32)
        weight = np.zeros((final_H, final_W, C), dtype=np.float32)
        mask = create_symmetric_2dmask(hres, wres, overlap)[..., None]
        idx = 0
        for i in range(zu):
            for j in range(zv):
                top = i * step_h
                left = j * step_w
                # Each patch is assumed to be (hres, wres, C)
                out[top:top+hres, left:left+wres, :] += patches[idx] * mask
                weight[top:top+hres, left:left+wres, :] += mask
                idx += 1
        out /= np.maximum(weight, 1e-8)
        return out[:H, :W, :]
