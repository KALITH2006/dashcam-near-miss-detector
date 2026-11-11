"""Monocular distance, trajectory prediction and collision-risk scoring.

World frame (metres, origin = camera on the motorcycle):
    Z : distance BEHIND the bike along the road (Z > 0 for vehicles in view)
    X : lateral offset in image direction (X < 0 = image-left)

Pipeline per tracked vehicle
    1. Pinhole distance      Z = f * H_real / h_px ,  X = (cx - W/2) * Z / f
    2. Kalman filter (X, Z)  -> smoothed position and relative velocity (vX, vZ)
    3. TTC                   world TTC = (Z - L) / closing speed, fused with looming TTC = h / (dh/dt)
    4. Trajectory prediction constant-velocity extrapolation over `horizon_s`
    5. Collision geometry    REAR-END : reaches contact plane inside the ego corridor
                             SIDE     : alongside, lateral offset closing into the corridor
    6. Risk score (0-100)    weighted TTC + distance + path-intersection terms
"""
from dataclasses import dataclass, field
import numpy as np
from kalman import KalmanCV


@dataclass
class Assessment:
    tid: int
    cls: str
    X: float
    Z: float
    vX: float
    vZ: float
    ttc: float
    risk: float
    level: str
    ctype: str          # "REAR-END", "SIDE-LEFT", "SIDE-RIGHT", "NONE"
    will_collide: bool
    path: list = field(default_factory=list)   # predicted [(X, Z), ...]
    ready: bool = True


def level_of(risk, cfg):
    if risk >= cfg.th_danger:
        return "DANGER"
    if risk >= cfg.th_warning:
        return "WARNING"
    if risk >= cfg.th_caution:
        return "CAUTION"
    return "SAFE"


class RiskEngine:
    def __init__(self, cfg, frame_w, frame_h):
        self.cfg = cfg
        self.W, self.H = frame_w, frame_h
        self.f = (frame_w / 2) / np.tan(np.radians(cfg.hfov_deg) / 2)
        self.horizon_y = frame_h * cfg.horizon_ratio
        self.filters = {}
        self.smooth = {}
        self.peaks = {}
        self.streak = {}
        self.events = []

    # ---------------- geometry helpers ----------------
    def project(self, X, Z):
        """Ground-plane point (X, Z) -> image pixel."""
        Z = max(Z, 0.5)
        return (self.W / 2 + self.f * X / Z,
                self.horizon_y + self.f * self.cfg.cam_height / Z)

    def side_label(self, X):
        left_image = X < 0
        if self.cfg.rear_camera:           # camera looks backwards: mirrored
            return "SIDE-RIGHT" if left_image else "SIDE-LEFT"
        return "SIDE-LEFT" if left_image else "SIDE-RIGHT"

    # ---------------- main step ----------------
    def step(self, tracks, dt, t):
        cfg = self.cfg
        out = []
        alive = set()
        for tr in tracks:
            alive.add(tr.id)
            x1, y1, x2, y2 = tr.box
            h = max(y2 - y1, 1.0)
            Hreal = cfg.real_height.get(tr.cls, 1.5)
            Zm = self.f * Hreal / h
            Xm = ((x1 + x2) / 2 - self.W / 2) * Zm / self.f

            kf = self.filters.get(tr.id)
            if kf is None:
                kf = KalmanCV([Xm, Zm], q=6.0, r=1.0, vel_var=25.0)
                self.filters[tr.id] = kf
            else:
                kf.predict(dt)
            if tr.matched:
                # depth noise grows quadratically with range
                kf.update([Xm, Zm], r_diag=[(0.04 * Zm) ** 2 + 0.01, (0.05 * Zm) ** 2 + 0.04])
            X, Z = kf.pos
            vX, vZ = np.clip(kf.vel, -cfg.max_rel_speed, cfg.max_rel_speed)
            Z = max(Z, 0.5)
            a = self._assess(tr, X, Z, vX, vZ, kf.updates)
            out.append(a)
            self._track_peak(a, t)

        for tid in list(self.filters):       # forget vanished tracks, flush their events
            if tid not in alive:
                self._flush(tid)
                self.filters.pop(tid, None)
                self.smooth.pop(tid, None)
                self.streak.pop(tid, None)
        return out

    # ---------------- risk model ----------------
    def _assess(self, tr, X, Z, vX, vZ, n_upd):
        cfg = self.cfg
        if n_upd < cfg.warmup_updates:
            return Assessment(tr.id, tr.cls, X, Z, vX, vZ, np.inf, 0.0, "SAFE", "NONE",
                              False, [], ready=False)
        Wc, Lc, T = cfg.half_width_c, cfg.length_c, cfg.horizon_s
        closing = -vZ                                   # > 0 : vehicle approaching

        # --- TTC: world model + looming (bbox growth) ---
        ttc_w = (max(Z - Lc, 0.0) / closing) if closing > 0.3 else np.inf
        h = max(tr.box[3] - tr.box[1], 1.0)
        ttc_l = h / tr.h_rate if tr.h_rate > 1.0 else np.inf
        if np.isfinite(ttc_w) and np.isfinite(ttc_l):
            ttc = 0.6 * ttc_w + 0.4 * min(ttc_l, 30.0)
        elif np.isfinite(ttc_w):
            ttc = ttc_w
        elif ttc_l < 4.0:
            ttc = ttc_l
        else:
            ttc = np.inf

        # --- REAR-END: will it reach the contact plane inside the ego corridor? ---
        rear_path, t_rear = 0.0, np.inf
        if closing > 0.3:
            t_c = max((Z - Lc) / closing, 0.0)
            if t_c <= T:
                Xc = X + vX * t_c
                rear_path = float(np.clip(1.0 - (abs(Xc) - Wc) / (1.5 * Wc), 0.0, 1.0))
                t_rear = ttc if np.isfinite(ttc) else t_c

        # --- SIDE: vehicle alongside / drifting laterally into the corridor ---
        side_path, t_side = 0.0, np.inf
        lat_close = -np.sign(X) * vX                   # > 0 : moving toward ego lane
        if Z < cfg.side_zone_z:
            if abs(X) <= Wc and Z <= Lc + 4.0:         # already overlapping
                side_path, t_side = 1.0, 0.0
            elif abs(X) > Wc and lat_close > cfg.lat_close_min:
                t_l = (abs(X) - Wc) / lat_close
                Zl = Z + vZ * t_l
                if t_l <= T and -4.0 <= Zl <= Lc + 4.0:
                    side_path, t_side = 0.9, t_l

        if side_path > rear_path:
            ctype = self.side_label(X)
            path_s, ttc_eff = side_path, t_side
        elif rear_path > 0:
            if cfg.rear_camera:
                ctype = "REAR-END"
            else:                                      # forward camera: ahead vs cutting in
                ctype = "FRONTAL" if abs(X) <= Wc else "CUT-IN"
            path_s, ttc_eff = rear_path, t_rear
        else:
            ctype, path_s, ttc_eff = "NONE", 0.0, ttc

        ttc_s = float(np.clip((cfg.ttc_max - ttc_eff) / (cfg.ttc_max - 1.0), 0, 1)) \
            if np.isfinite(ttc_eff) else 0.0
        dist_s = float(np.clip(1.0 - (Z - 2.0) / (cfg.z_max - 2.0), 0, 1))
        raw = 100.0 * (0.5 * ttc_s + 0.2 * dist_s + 0.3 * path_s) * (0.5 + 0.5 * path_s)
        if closing <= 0.3 and ctype == "NONE":         # receding / keeping distance
            raw *= 0.3
        prev = self.smooth.get(tr.id, raw)
        risk = cfg.risk_ema * raw + (1 - cfg.risk_ema) * prev
        self.smooth[tr.id] = risk

        # --- predicted path for visualisation ---
        path = []
        for s in np.arange(0.0, T + 1e-6, cfg.pred_step):
            pz = Z + vZ * s
            if pz < 0.8:
                break
            path.append((X + vX * s, pz))

        will = ctype != "NONE" and path_s > 0.5 and np.isfinite(ttc_eff) and ttc_eff <= T
        return Assessment(tr.id, tr.cls, X, Z, vX, vZ, float(ttc_eff), float(risk),
                          level_of(risk, cfg), ctype, bool(will), path)

    # ---------------- near-miss event log ----------------
    def _track_peak(self, a, t):
        if not a.ready:
            return
        # a threat only counts once risk has stayed high for `event_sustain` updates in a row
        s = self.streak.get(a.tid, 0) + 1 if a.risk >= self.cfg.event_min_peak else 0
        self.streak[a.tid] = s
        if s < self.cfg.event_sustain:
            return
        p = self.peaks.get(a.tid)
        if p is None or a.risk > p["peak"]:
            self.peaks[a.tid] = dict(tid=a.tid, cls=a.cls, peak=a.risk, t=t, ctype=a.ctype,
                                     Z=a.Z, ttc=a.ttc, min_ttc=min(a.ttc, p["min_ttc"]) if p else a.ttc)
        else:
            p["min_ttc"] = min(p["min_ttc"], a.ttc)

    def _flush(self, tid):
        p = self.peaks.pop(tid, None)
        if p and p["peak"] >= self.cfg.event_min_peak:
            p["event"] = "COLLISION-IMMINENT" if p["peak"] >= self.cfg.th_danger else "NEAR-MISS"
            self.events.append(p)

    def flush_all(self):
        for tid in list(self.peaks):
            self._flush(tid)
        self.events.sort(key=lambda e: e["t"])
