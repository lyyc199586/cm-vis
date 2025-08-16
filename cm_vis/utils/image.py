import numpy as np

def crop(
    img: np.ndarray,
    thresh: float = 0.95,
    margin: int = 0,
    verbose: bool = False
) -> np.ndarray:
    """
    Crop the image by removing nearly-white or transparent borders.

    Parameters
    ----------
    img : np.ndarray
        Input image array of shape (H, W, C), with 3 or 4 channels (RGB or RGBA).
    thresh : float, optional
        Threshold (0-1) to consider a pixel as background. Default is 0.95.
    margin : int, optional
        Number of pixels to retain around the detected content region. Default is 0.
    verbose : bool, optional
        If True, print warning when nothing is cropped.

    Returns
    -------
    np.ndarray
        Cropped image containing only the relevant content.
    """
    if img.dtype != np.float32 and img.dtype != np.float64:
        img_norm = img / 255.0
    else:
        img_norm = img

    if img.shape[2] == 4:
        # Use RGB * alpha if alpha channel exists
        gray = np.mean(img_norm[..., :3], axis=2) * img_norm[..., 3]
    else:
        gray = np.mean(img_norm[..., :3], axis=2)

    mask = gray < thresh

    if not np.any(mask):
        if verbose:
            print("Warning: No pixels below threshold, returning original image.")
        return img

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    y0 = max(y0 - margin, 0)
    x0 = max(x0 - margin, 0)
    y1 = min(y1 + margin, img.shape[0])
    x1 = min(x1 + margin, img.shape[1])

    return img[y0:y1, x0:x1]
