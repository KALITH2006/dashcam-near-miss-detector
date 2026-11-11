"""Synthetic rear-view traffic video generator (pinhole-consistent) + ground-truth boxes.

Scenarios (relative to the motorcycle):
  A  car in ego lane closes in fast, brakes late          -> REAR-END near-miss
  B  car in left adjacent lane overtakes and cuts in      -> SIDE threat
  C  car in right lane keeps a safe gap                   -> SAFE
  D  truck far behind in ego lane                         -> SAFE
Use only for testing the pipeline; real footage is processed with --detector yolo.
"""
import argparse
import json
import cv2
import numpy as np

W, H, FPS, DUR = 960, 540, 20, 20
HFOV, CAM_H, HORIZON = 90.0, 1.0, 0.45
F = (W / 2) / np.tan(np.radians(HFOV) / 2)
HY = H * HORIZON
SIZE = {"car": (1.8, 1.5), "truck": (2.5, 3.0), "motorcycle": (0.8, 1.4)}   # width, height (m)


def scen_a(t):                      # in-lane rear-end approach
    z0, v, tb, a = 50.0, 6.0, 6.0, 2.5
    if t <= tb:
        return 0.0, z0 - v * t
    td = min(t - tb, v / a)
    return 0.0, z0 - v * tb - (v * td - 0.5 * a * td * td)


def scen_b(t):                      # left-lane overtake with cut-in
    z = 40.0 - 4.0 * max(t - 2.0, 0.0) if t >= 2 else 40.0
    x = -3.5
    if t > 9.0:
        x = -3.5 + min(1.2 * (t - 9.0), 2.3)
    return x, z


def scen_c(t):
    return 3.5, 25.0 + 0.3 * np.sin(t / 3)


def scen_d(t):
    return 0.0, 70.0 - 0.5 * t


VEHICLES = [("A", "car", scen_a, (200, 60, 50)), ("B", "car", scen_b, (60, 90, 200)),
            ("C", "motorcycle", scen_c, (60, 160, 60)), ("D", "truck", scen_d, (190, 190, 190))]


def proj(X, Z):
    return W / 2 + F * X / Z, HY + F * CAM_H / Z


def render(t):
    img = np.zeros((H, W, 3), np.uint8)
    img[:int(HY)] = (200, 170, 130)
    img[int(HY):] = (70, 120, 70)
    road = np.array([proj(x, z) for x, z in [(-7, 3), (-7, 120), (7, 120), (7, 3)]], np.int32)
    cv2.fillPoly(img, [road], (90, 90, 95))
    for lx in (-5.25, -1.75, 1.75, 5.25):
        for z in np.arange(3, 120, 4):
            if int(z / 4) % 2 == 0:
                cv2.line(img, tuple(map(int, proj(lx, z))), tuple(map(int, proj(lx, z + 3))),
                         (230, 230, 230), max(1, int(F * 0.08 / z)))
    gt = []
    for _, cls, fn, col in sorted(VEHICLES, key=lambda v: -v[2](t)[1]):     # far first
        X, Z = fn(t)
        if Z < 2.8:
            continue
        w, h = SIZE[cls]
        xb, yb = proj(X, Z)
        pw, ph = F * w / Z, F * h / Z
        x1, x2, y2, y1 = xb - pw / 2, xb + pw / 2, yb, yb - ph
        cv2.ellipse(img, (int(xb), int(yb)), (int(pw * .55), max(1, int(ph * .06))), 0, 0, 360, (40, 40, 40), -1)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, -1)
        cv2.rectangle(img, (int(x1 + pw * .12), int(y1 + ph * .12)), (int(x2 - pw * .12), int(y1 + ph * .45)),
                      (40, 40, 50), -1)
        for lx in (x1 + pw * .05, x2 - pw * .2):
            cv2.rectangle(img, (int(lx), int(y2 - ph * .3)), (int(lx + pw * .15), int(y2 - ph * .18)), (0, 0, 220), -1)
        gx1, gy1, gx2, gy2 = max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)
        if gx2 - gx1 > 4 and gy2 - gy1 > 4:
            gt.append([float(gx1), float(gy1), float(gx2), float(gy2), cls])
    return img, gt


def generate(out_video, out_gt):
    vw = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames = {}
    for i in range(FPS * DUR):
        img, gt = render(i / FPS)
        vw.write(img)
        frames[str(i)] = gt
    vw.release()
    with open(out_gt, "w") as f:
        json.dump({"fps": FPS, "frames": frames}, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo/demo.mp4")
    ap.add_argument("--gt", default="demo/demo_gt.json")
    a = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    generate(a.out, a.gt)
    print("wrote", a.out, a.gt)
