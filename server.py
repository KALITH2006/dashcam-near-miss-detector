"""Near-Miss Detector web app.   python server.py   ->  http://localhost:5000
Flask backend + hand-built HTML/JS front end (web/index.html). Camera set-up is automatic."""
import os
import threading
import uuid

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

import calibrate
import pipeline
from config import Config

ROOT = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(ROOT, "jobs")
app = Flask(__name__, static_folder=None)
jobs = {}                      # id -> {state, progress, stage, error}
run_lock = threading.Lock()    # one analysis at a time (YOLO is heavy on CPU / RAM)


def _worker(jid, src, gt, demo):
    job = jobs[jid]
    out = os.path.join(JOBS_DIR, jid)
    try:
        with run_lock:
            if demo:
                cfg, detector, stride, cal = Config(), "sim", 1, None
            else:
                job.update(state="running", stage="Setting up the camera automatically...", progress=0.0)
                cal = calibrate.auto_calibrate(src)
                cfg = Config(hfov_deg=cal["hfov"], cam_height=cal["cam_h"], horizon_ratio=cal["horizon"],
                             rear_camera=False)
                detector, stride = "yolo", cal["stride"]
            job.update(state="running", stage="Finding and following vehicles...", progress=0.0)
            s = pipeline.run(src, out, cfg, detector, "yolov8n.pt", gt, stride,
                             progress=lambda p: job.update(progress=float(p)))
            job.update(state="done", progress=1.0, stage="Done", cal=cal, summary=s,
                       camera="rear" if demo else "front")
    except Exception as e:                      # surfaced to the UI
        job.update(state="error", error=str(e))


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 2)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


@app.get("/")
def index():
    return send_from_directory(os.path.join(ROOT, "web"), "index.html")


@app.post("/api/start")
def start():
    jid = uuid.uuid4().hex[:10]
    out = os.path.join(JOBS_DIR, jid)
    os.makedirs(out, exist_ok=True)
    jobs[jid] = dict(state="queued", progress=0.0, stage="Waiting...")
    if request.args.get("demo"):
        import demo_video
        src, gt = os.path.join(out, "demo.mp4"), os.path.join(out, "demo_gt.json")
        demo_video.generate(src, gt)
        threading.Thread(target=_worker, args=(jid, src, gt, True), daemon=True).start()
    else:
        f = request.files.get("video")
        if not f:
            return jsonify(error="No video received"), 400
        src = os.path.join(out, "input.mp4")
        f.save(src)
        threading.Thread(target=_worker, args=(jid, src, None, False), daemon=True).start()
    return jsonify(id=jid)


@app.get("/api/status/<jid>")
def status(jid):
    j = jobs.get(jid) or abort(404)
    return jsonify({k: j.get(k) for k in ("state", "progress", "stage", "error")})


@app.get("/api/result/<jid>")
def result(jid):
    j = jobs.get(jid) or abort(404)
    if j.get("state") != "done":
        abort(409)
    out = os.path.join(JOBS_DIR, jid)
    s = j["summary"]
    ev = pd.read_csv(os.path.join(out, "events.csv"))
    tr = pd.read_csv(os.path.join(out, "tracks.csv"))
    samples = []
    if len(tr):
        for t, g in tr.groupby("time_s"):
            w = g.loc[g["risk"].idxmax()]
            samples.append([round(float(t), 2), round(float(w["risk"]), 1), w["level"], w["collision_type"],
                            w["class"], _clean(w["Z_m"]), _clean(w["ttc_s"]), int(len(g))])
    events = [{k: _clean(v) for k, v in r.items()} for r in ev.to_dict("records")]
    cal = j.get("cal")
    # compact per-vehicle positions for the radar view and the per-vehicle threat timeline
    tracks = [[round(float(a), 2), int(b), c, round(float(d), 1), round(float(e), 1), round(float(f), 0)]
              for a, b, c, d, e, f in tr[["time_s", "track_id", "class", "X_m", "Z_m", "risk"]].itertuples(index=False)]
    return jsonify(duration=s["frames"] / s["fps"], fps=s["fps"], events=events, samples=samples,
                   tracks=tracks, camera=j.get("camera", "front"), calibration=cal, video=f"/video/{jid}")


@app.get("/video/<jid>")
def video(jid):
    p = os.path.join(JOBS_DIR, jid, "annotated.mp4")
    if not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="video/mp4", conditional=True)       # conditional = range requests / seeking


@app.get("/events/<jid>.csv")
def events_csv(jid):
    p = os.path.join(JOBS_DIR, jid, "events.csv")
    if not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype="text/csv", as_attachment=True, download_name="near_miss_events.csv")


if __name__ == "__main__":
    os.makedirs(JOBS_DIR, exist_ok=True)
    print("Near-Miss Detector -> http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)
