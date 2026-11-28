"""
Importance map: where likeness matters most (faces, eyes, mouth), as 0..1 per pixel.

Sources, combined with max():
* face detection: OpenCV's YuNet detector (models/face_detection_yunet_2023mar.onnx,
  MIT, from opencv_zoo). Finds tilted and partly turned faces and gives five
  landmarks per face. Each face gets a soft ellipse at `FACE`, eyes and mouth more.
  Needs opencv-python; without it the map comes only from the painted mask.
* a painted mask: grayscale image the size of the source photo, white = important
  (the brush in the preview UI).

The map is cropped and resized exactly like the target, so it lines up with it.
String art uses it as an extra error weight: weight *= 1 + strength * map. Flow hatch and
scribble don't optimise an error, so there it means finer detail: hatch directions and tone
are smoothed less, scribble keeps weaker edges, traces them first and packs contours closer.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageFilter

MODEL = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
FACE, MOUTH, EYES = 0.5, 0.8, 1.0


def detect_faces(rgb, score=0.7):
    """rgb: PIL image. Returns a list of dicts {box: (x, y, w, h), eyes: [(x, y), (x, y)],
    nose: (x, y), mouth: [(x, y), (x, y)], score} in image pixels. [] without OpenCV."""
    try:
        import cv2
    except ImportError:
        return []
    a = np.asarray(rgb.convert("RGB"))[:, :, ::-1]
    sc = min(1.0, 960 / max(a.shape[:2]))
    if sc < 1:
        a = cv2.resize(a, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    a = np.ascontiguousarray(a)
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (a.shape[1], a.shape[0]), score)
    _, faces = det.detect(a)
    out = []
    for f in ([] if faces is None else faces):
        p = [float(v) / sc for v in f[:14]]
        out.append({"box": tuple(p[:4]), "eyes": [(p[4], p[5]), (p[6], p[7])], "nose": (p[8], p[9]),
                    "mouth": [(p[10], p[11]), (p[12], p[13])], "score": float(f[14])})
    return out


def _ellipse(shape, cx, cy, rx, ry):
    h, w = shape
    y0, y1 = max(0, int(cy - ry) - 1), min(h, int(cy + ry) + 2)
    x0, x1 = max(0, int(cx - rx) - 1), min(w, int(cx + rx) + 2)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return (slice(y0, y1), slice(x0, x1)), ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def _paint(m, value, cx, cy, rx, ry):
    sl, inside = _ellipse(m.shape, cx, cy, max(rx, 1), max(ry, 1))
    m[sl][inside] = np.maximum(m[sl][inside], value)


def source_map(image_path, face=True, mask_path=None):
    """Importance at source-image resolution (0..1) and a summary of what was found."""
    img = ImageOps.exif_transpose(Image.open(image_path))
    m = np.zeros((img.height, img.width), np.float32)
    info = {"faces": 0, "painted": False}
    if face:
        faces = detect_faces(img)
        info["faces"] = len(faces)
        for f in faces:
            x, y, w, h = f["box"]
            _paint(m, FACE, x + w / 2, y + h * 0.55, w * 0.55, h * 0.62)
            (ax, ay), (bx, by) = f["eyes"]
            d = float(np.hypot(bx - ax, by - ay))
            for ex, ey in f["eyes"]:                       # eye + brow
                _paint(m, EYES, ex, ey - 0.1 * d, 0.38 * d, 0.3 * d)
            (ax, ay), (bx, by) = f["mouth"]
            md = float(np.hypot(bx - ax, by - ay))
            _paint(m, MOUTH, (ax + bx) / 2, (ay + by) / 2, 0.75 * md, 0.4 * md)
    if mask_path and Path(mask_path).exists():
        pm = Image.open(mask_path).convert("L").resize(img.size, Image.Resampling.BILINEAR)
        m = np.maximum(m, np.asarray(pm, np.float32) / 255.0)
        info["painted"] = True
    if m.any():
        blur = max(2, min(img.size) // 100)                # soft edges, no visible seams
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)),
                       np.float32) / 255.0
    return m, info


def importance_map(image_path, box, size_px, face=True, mask_path=None):
    """Importance on the solver grid: crop `box` (x, y, w, h source pixels) and resize to
    size_px x size_px, exactly like the target."""
    return importance_grid(image_path, box, (size_px, size_px), face, mask_path)


def importance_grid(image_path, box, shape, face=True, mask_path=None):
    """Importance on any H x W grid: crop `box` (x, y, w, h source pixels, None = whole
    photo) and resize to shape = (H, W)."""
    m, info = source_map(image_path, face, mask_path)
    im = Image.fromarray((m * 255).astype(np.uint8))
    if box is not None:
        x, y, w, h = box
        im = im.crop((x, y, x + w, y + h))
    im = im.resize((int(shape[1]), int(shape[0])), Image.Resampling.BILINEAR)
    return np.asarray(im, np.float32) / 255.0, info
