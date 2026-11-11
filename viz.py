"""Overlay drawing: boxes, predicted paths, HUD, and a bird's-eye-view mini-map."""
import cv2
import numpy as np

COLORS = {  # BGR
    "SAFE": (80, 200, 80), "CAUTION": (0, 215, 255),
    "WARNING": (0, 140, 255), "DANGER": (40, 40, 255),
}


def _text(img, s, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_frame(frame, tracks, assessments, engine, t):
    img = frame.copy()
    by_id = {a.tid: a for a in assessments}
    for tr in tracks:
        a = by_id.get(tr.id)
        if a is None:
            continue
        col = COLORS[a.level]
        x1, y1, x2, y2 = map(int, tr.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        # predicted trajectory projected back onto the road
        if a.ready and len(a.path) > 1:
            pts = np.array([engine.project(px, pz) for px, pz in a.path], np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, col, 2, cv2.LINE_AA)
        ttc = f"{a.ttc:.1f}s" if np.isfinite(a.ttc) else "--"
        lab = f"#{a.tid} {a.cls} {a.Z:.0f}m TTC {ttc} R{a.risk:.0f}"
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, lab, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        if a.will_collide:
            _text(img, a.ctype, (x1, y2 + 16), 0.5, col, 1)

    _bev(img, assessments, engine)
    _hud(img, assessments, t)
    return img


def _hud(img, assessments, t):
    ready = [a for a in assessments if a.ready]
    worst = max(ready, key=lambda a: a.risk) if ready else None
    risk = worst.risk if worst else 0.0
    level = worst.level if worst else "SAFE"
    col = COLORS[level]
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (25, 25, 25), -1)
    _text(img, f"t={t:5.1f}s", (10, 23), 0.6)
    cv2.rectangle(img, (110, 8), (310, 26), (70, 70, 70), -1)
    cv2.rectangle(img, (110, 8), (110 + int(2 * risk), 26), col, -1)
    _text(img, f"RISK {risk:3.0f}  {level}", (320, 24), 0.6, col, 2)
    if worst and level in ("WARNING", "DANGER"):
        msg = f"{worst.ctype} THREAT  #{worst.tid}"
        _text(img, msg, (img.shape[1] - 330, 24), 0.6, col, 2)
        if level == "DANGER" and int(t * 4) % 2 == 0:
            cv2.rectangle(img, (2, 36), (img.shape[1] - 3, img.shape[0] - 3), col, 4)


def _bev(img, assessments, engine, pw=190, ph=300, zmax=40.0, xmax=6.0):
    H, W = img.shape[:2]
    x0, y0 = W - pw - 10, 44
    panel = np.full((ph, pw, 3), 35, np.uint8)
    sx, sz = pw / (2 * xmax), (ph - 30) / zmax
    to_px = lambda X, Z: (int(pw / 2 + X * sx), int(ph - 20 - Z * sz))
    for lx in (-5.25, -1.75, 1.75, 5.25):
        cv2.line(panel, to_px(lx, 0), to_px(lx, zmax), (90, 90, 90), 1)
    for z in range(0, int(zmax) + 1, 10):
        cv2.line(panel, (0, to_px(0, z)[1]), (6, to_px(0, z)[1]), (120, 120, 120), 1)
        cv2.putText(panel, f"{z}m", (8, to_px(0, z)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    c = engine.cfg
    cv2.rectangle(panel, to_px(-c.half_width_c, c.length_c), to_px(c.half_width_c, 0), (60, 60, 140), 1)
    cv2.rectangle(panel, to_px(-0.4, 0.5), to_px(0.4, -1.0), (255, 255, 255), -1)   # ego bike
    for a in assessments:
        col = COLORS[a.level]
        if a.ready and len(a.path) > 1:
            pts = np.array([to_px(px, pz) for px, pz in a.path], np.int32)
            cv2.polylines(panel, [pts.reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
        w = c.real_width.get(a.cls, 1.8) / 2
        cv2.rectangle(panel, to_px(a.X - w, a.Z + 2.0), to_px(a.X + w, a.Z), col, -1)
    cv2.putText(panel, "BEV (rear view)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    if y0 + ph < H and x0 > 0:
        img[y0:y0 + ph, x0:x0 + pw] = cv2.addWeighted(img[y0:y0 + ph, x0:x0 + pw], 0.2, panel, 0.8, 0)
