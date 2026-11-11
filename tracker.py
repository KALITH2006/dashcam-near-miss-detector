"""SORT-style multi-object tracker: Kalman (image space) + IoU + Hungarian matching."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from kalman import KalmanCV


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0), max(y2 - y1, 1.0)])


def cxcywh_to_xyxy(s):
    cx, cy, w, h = s
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def iou_matrix(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


class Track:
    def __init__(self, tid, box, cls, conf):
        self.id = tid
        self.cls = cls
        self.conf = conf
        self.kf = KalmanCV(xyxy_to_cxcywh(box), q=400.0, r=9.0, vel_var=50.0)
        self.hits = 1
        self.misses = 0
        self.age = 0
        self.matched = True

    @property
    def box(self):
        return cxcywh_to_xyxy(self.kf.pos)

    @property
    def h_rate(self):
        """Image-space growth rate of bbox height (px/s) -> used for looming TTC."""
        return float(self.kf.vel[3])

    def predict(self, dt):
        self.kf.predict(dt)
        self.kf.x[2:4] = np.maximum(self.kf.x[2:4], 1.0)
        self.age += 1
        self.matched = False

    def update(self, box, cls, conf):
        self.kf.update(xyxy_to_cxcywh(box))
        self.hits += 1
        self.misses = 0
        self.matched = True
        if conf >= self.conf:
            self.cls, self.conf = cls, conf


class Tracker:
    def __init__(self, iou_thr=0.2, max_age=12, min_hits=3):
        self.iou_thr, self.max_age, self.min_hits = iou_thr, max_age, min_hits
        self.tracks = []
        self._next = 1

    def update(self, dets, dt):
        for t in self.tracks:
            t.predict(dt)
        boxes = [d[:4] for d in dets]
        pred = [t.box for t in self.tracks]
        iou = iou_matrix(pred, boxes)
        matched_t, matched_d = set(), set()
        if iou.size:
            r, c = linear_sum_assignment(1.0 - iou)
            for i, j in zip(r, c):
                if iou[i, j] >= self.iou_thr:
                    self.tracks[i].update(boxes[j], dets[j][5], dets[j][4])
                    matched_t.add(i)
                    matched_d.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks.append(Track(self._next, d[:4], d[5], d[4]))
                self._next += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses <= 3]
