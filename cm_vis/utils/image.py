import numpy as np

def crop(
    img: np.ndarray,
    *,
    alpha_thresh: float = 1/255,   # Content if alpha > threshold
    white_thresh: float = 0.95,    # For white background: content = gray < white_thresh
    black_thresh: float = 0.05,    # For black background: content = gray > black_thresh
    margin: int = 0,
    verbose: bool = True
) -> np.ndarray:
    """
    cropping function:
    - If image has an alpha channel, use it to detect content (transparent = background).
    - If no alpha channel, automatically detect black/white background from borders.
    - Supports margin, preserves original dtype, and handles edge cases.

    Parameters
    ----------
    img : np.ndarray
        Input image of shape (H, W, C), with C=3 (RGB) or C=4 (RGBA).
    alpha_thresh : float, optional
        Threshold for alpha channel to consider a pixel as content. Default: 1/255.
    white_thresh : float, optional
        Threshold for content detection when background is white. Default: 0.95.
    black_thresh : float, optional
        Threshold for content detection when background is black. Default: 0.05.
    margin : int, optional
        Extra pixels to keep around the detected bounding box. Default: 0.
    verbose : bool, optional
        If True, print diagnostic messages.

    Returns
    -------
    np.ndarray
        Cropped image with the same dtype as the input.
    """
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("img must be (H, W, C) with C=3 (RGB) or C=4 (RGBA).")
    H, W, C = img.shape
    orig_dtype = img.dtype

    # Normalize to [0,1] for detection
    if img.dtype in (np.float32, np.float64):
        img_norm = img
    else:
        img_norm = img.astype(np.float32) / 255.0

    rgb = img_norm[..., :3]
    gray = rgb.mean(axis=2)

    if C == 4:
        # Use alpha channel if available
        alpha = img_norm[..., 3]
        content_mask = alpha > alpha_thresh
    else:
        # Estimate background brightness from border pixels (median is more robust)
        border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        bg_med = float(np.median(border))
        if bg_med > 0.5:
            # White background → content is darker
            content_mask = gray < white_thresh
        else:
            # Black background → content is brighter
            content_mask = gray > black_thresh

    # Handle corner cases
    if not np.any(content_mask):
        if verbose:
            print("crop_smart: no content detected; returning original image.")
        return img
    if np.all(content_mask):
        if verbose:
            print("crop_smart: content covers entire image; returning original image.")
        return img

    coords = np.argwhere(content_mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    if margin:
        y0 = max(y0 - margin, 0)
        x0 = max(x0 - margin, 0)
        y1 = min(y1 + margin, H)
        x1 = min(x1 + margin, W)

    cropped = img[y0:y1, x0:x1]
    
    return cropped
