"""Automatic camera set-up: no manual FOV / height / horizon / frame-skip needed.

* horizon  : median vanishing-point row of road lane lines found by Hough lines on sample frames
             (falls back to 0.5 of frame height if not enough lines are found)
* hfov     : typical dash-cam value (video files carry no lens data)
* cam_h    : typical dash-cam height
* stride   : chosen from fps so ~15 frames per second are analysed
"""
import cv2
import numpy as np


def _vp_rows(frame):
    h, w = frame.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    e = cv2.Canny(g, 60, 150)
    mask = np.zeros_like(e)
    mask[int(h * 0.45):int(h * 0.85)] = 255          # road area, below horizon, above bonnet
    lines = cv2.HoughLinesP(cv2.bitwise_and(e, mask), 1, np.pi / 180, 40,
                            minLineLength=int(w * 0.08), maxLineGap=int(w * 0.02))
    if lines is None:
        return []
    pos, neg = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        if 0.35 < abs(m) < 3.0:
            (pos if m > 0 else neg).append((m, y1 - m * x1))      # y = m x + b
    rows = []
    for m1, b1 in pos[:60]:
        for m2, b2 in neg[:60]:
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            if 0.2 * w < x < 0.8 * w and 0.15 * h < y < 0.7 * h:
                rows.append(y / h)
    return rows


def auto_calibrate(path, samples=12):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    meds = []
    for k in range(samples):
        if n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / samples))
        ok, f = cap.read()
        if not ok:
            continue
        r = _vp_rows(f)
        if len(r) >= 5:
            meds.append(float(np.median(r)))
    cap.release()
    horizon = float(np.clip(np.median(meds), 0.3, 0.6)) if len(meds) >= 3 else 0.5
    return dict(hfov=90.0, cam_h=1.3, horizon=round(horizon, 2),
                stride=max(1, round(fps / 15.0)), fps=fps, horizon_found=len(meds) >= 3)
