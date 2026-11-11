"""Vehicle detectors.

YoloDetector : real detector (Ultralytics YOLO, COCO classes).
SimDetector  : replays ground-truth boxes (+ noise / dropouts) from the synthetic demo
               so the whole pipeline can be tested without a trained model or real video.

A detection is a tuple (x1, y1, x2, y2, conf, class_name).
"""
import json
import numpy as np

COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class YoloDetector:
    def __init__(self, weights="yolov8n.pt", conf=0.35, imgsz=960, device=None):
        from ultralytics import YOLO  # imported lazily so the sim mode needs no torch
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def __call__(self, frame, frame_idx=0):
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 classes=list(COCO_VEHICLES), device=self.device,
                                 verbose=False)[0]
        dets = []
        if res.boxes is None:
            return dets
        for b, c, k in zip(res.boxes.xyxy.cpu().numpy(),
                           res.boxes.conf.cpu().numpy(),
                           res.boxes.cls.cpu().numpy()):
            dets.append((*map(float, b), float(c), COCO_VEHICLES.get(int(k), "car")))
        return dets


class SimDetector:
    def __init__(self, gt_json, jitter=0.015, dropout=0.03, seed=0):
        with open(gt_json) as f:
            self.gt = json.load(f)["frames"]
        self.jitter, self.dropout = jitter, dropout
        self.rng = np.random.default_rng(seed)

    def __call__(self, frame, frame_idx=0):
        dets = []
        for x1, y1, x2, y2, cls in self.gt.get(str(frame_idx), []):
            if self.rng.random() < self.dropout:
                continue
            w, h = x2 - x1, y2 - y1
            n = self.rng.normal(0, self.jitter, 4) * np.array([w, h, w, h])
            dets.append((x1 + n[0], y1 + n[1], x2 + n[2], y2 + n[3], 0.9, cls))
        return dets
