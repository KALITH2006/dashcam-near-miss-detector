"""Command-line entry point.

Examples
  python main.py --demo                                   # synthetic demo, no model/video needed
  python main.py --source ride.mp4 --detector yolo        # real video with YOLO
  python main.py --source ride.mp4 --weights best.pt --hfov 110 --show
"""
import argparse
import json
import os

from config import Config
import pipeline


def main():
    ap = argparse.ArgumentParser(description="Near-miss & collision risk prediction for motorcycles")
    ap.add_argument("--source", help="input video path")
    ap.add_argument("--demo", action="store_true", help="generate + process a synthetic demo video")
    ap.add_argument("--detector", choices=["yolo", "sim"], default="yolo")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--gt", help="ground-truth json (sim detector only)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--hfov", type=float, default=90.0, help="camera horizontal FOV in degrees")
    ap.add_argument("--cam-height", type=float, default=1.0)
    ap.add_argument("--horizon", type=float, default=0.45, help="horizon row / frame height")
    ap.add_argument("--front-camera", action="store_true", help="camera faces forward (default: rear)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--show", action="store_true", help="live preview window (ESC to quit)")
    a = ap.parse_args()

    cfg = Config(hfov_deg=a.hfov, cam_height=a.cam_height, horizon_ratio=a.horizon,
                 rear_camera=not a.front_camera)
    source, detector, gt = a.source, a.detector, a.gt
    if a.demo:
        import demo_video
        os.makedirs("demo", exist_ok=True)
        source, gt = "demo/demo.mp4", "demo/demo_gt.json"
        demo_video.generate(source, gt)
        detector = "sim"
        cfg = Config()
    if not source:
        ap.error("provide --source <video> or --demo")

    s = pipeline.run(source, a.out, cfg, detector, a.weights, gt, a.stride, a.show,
                     progress=lambda p: print(f"\r{p * 100:5.1f}%", end="", flush=True))
    print("\n" + json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
