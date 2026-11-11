"""End-to-end pipeline: video -> detect -> track -> predict -> risk -> annotated video + reports."""
import csv
import json
import os
import shutil
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from detector import YoloDetector, SimDetector
from tracker import Tracker
from risk import RiskEngine
from viz import draw_frame, COLORS


def build_detector(kind, cfg, weights="yolov8n.pt", gt=None):
    if kind == "sim":
        if not gt:
            raise ValueError("--gt <json> is required with --detector sim")
        return SimDetector(gt)
    return YoloDetector(weights, conf=cfg.conf, imgsz=cfg.imgsz)


def run(source, out_dir="outputs", cfg=None, detector="yolo", weights="yolov8n.pt",
        gt=None, stride=1, show=False, progress=None, max_frames=None):
    cfg = cfg or Config()
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    dt = stride / fps

    det = build_detector(detector, cfg, weights, gt)
    tracker = Tracker(cfg.iou_match, cfg.max_age, cfg.min_hits)
    engine = RiskEngine(cfg, W, H)

    raw_path = os.path.join(out_dir, "annotated_raw.mp4")
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (W, H))

    rows, timeline = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        if idx % stride == 0:
            t = idx / fps
            dets = det(frame, idx)
            tracks = tracker.update(dets, dt)
            assess = engine.step(tracks, dt, t)
            writer.write(draw_frame(frame, tracks, assess, engine, t)) if not show else None
            if show:
                vis = draw_frame(frame, tracks, assess, engine, t)
                writer.write(vis)
                cv2.imshow("Near-miss AI", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            ready = [a for a in assess if a.ready]
            timeline.append((t, max((a.risk for a in ready), default=0.0)))
            for a in ready:
                rows.append([idx, round(t, 3), a.tid, a.cls, round(a.X, 2), round(a.Z, 2),
                             round(a.vX, 2), round(a.vZ, 2),
                             round(a.ttc, 2) if np.isfinite(a.ttc) else "",
                             round(a.risk, 1), a.level, a.ctype, int(a.will_collide)])
        idx += 1
        if progress and total:
            progress(min(idx / total, 1.0))
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    engine.flush_all()

    final = _to_h264(raw_path, os.path.join(out_dir, "annotated.mp4"))

    with open(os.path.join(out_dir, "tracks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "track_id", "class", "X_m", "Z_m", "vX_mps", "vZ_mps",
                    "ttc_s", "risk", "level", "collision_type", "will_collide"])
        w.writerows(rows)
    with open(os.path.join(out_dir, "events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "track_id", "class", "event", "collision_type", "peak_risk",
                    "min_ttc_s", "dist_at_peak_m"])
        for e in engine.events:
            w.writerow([round(e["t"], 2), e["tid"], e["cls"], e["event"], e["ctype"],
                        round(e["peak"], 1), round(e["min_ttc"], 2) if np.isfinite(e["min_ttc"]) else "",
                        round(e["Z"], 1)])
    _plot_timeline(timeline, rows, cfg, os.path.join(out_dir, "risk_timeline.png"))

    summary = dict(frames=idx, fps=fps, resolution=[W, H],
                   max_risk=round(max((r for _, r in timeline), default=0.0), 1),
                   events=[{k: (round(v, 2) if isinstance(v, float) else v) for k, v in e.items()}
                           for e in engine.events],
                   video=final)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: None if o == np.inf else float(o))
    return summary


def _to_h264(src, dst):
    """Re-encode to H.264 so the video plays in browsers / Streamlit (if ffmpeg exists)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    if exe:
        r = subprocess.run([exe, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-preset", "veryfast", dst])
        if r.returncode == 0:
            os.remove(src)
            return dst
    os.replace(src, dst)
    return dst


def _plot_timeline(timeline, rows, cfg, path):
    if not timeline:
        return
    t = np.array([a for a, _ in timeline])
    r = np.array([b for _, b in timeline])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhspan(0, cfg.th_caution, color="#50c850", alpha=.15)
    ax.axhspan(cfg.th_caution, cfg.th_warning, color="#ffd700", alpha=.15)
    ax.axhspan(cfg.th_warning, cfg.th_danger, color="#ff8c00", alpha=.15)
    ax.axhspan(cfg.th_danger, 100, color="#ff2828", alpha=.15)
    ids = sorted({r_[2] for r_ in rows})
    for tid in ids:
        pts = [(r_[1], r_[9]) for r_ in rows if r_[2] == tid]
        if max(p[1] for p in pts) >= cfg.th_caution:
            ax.plot(*zip(*pts), lw=1.2, label=f"vehicle #{tid}")
    ax.plot(t, r, "k", lw=2, label="max risk")
    ax.set_xlabel("time (s)"); ax.set_ylabel("risk score"); ax.set_ylim(0, 100)
    ax.set_title("Collision-risk timeline"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
