"""Central configuration for the near-miss / collision-risk system."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- camera model (monocular, rear-facing camera mounted on the motorcycle) ----
    hfov_deg: float = 90.0        # horizontal field of view of the camera
    cam_height: float = 1.0       # camera height above road (m)
    horizon_ratio: float = 0.45   # horizon row as a fraction of frame height
    rear_camera: bool = True      # True: camera looks backwards (image-left = rider's right)

    # ---- detection ----
    conf: float = 0.35
    imgsz: int = 960
    classes: tuple = ("bicycle", "car", "motorcycle", "bus", "truck")

    # ---- tracking ----
    iou_match: float = 0.2
    max_age: int = 12             # frames a track survives without detections
    min_hits: int = 4             # detections before a track is "confirmed"

    # ---- prediction / risk ----
    horizon_s: float = 3.0        # prediction horizon (seconds)
    pred_step: float = 0.1        # sampling step of predicted path (s)
    ttc_max: float = 6.0          # TTC above this gives zero TTC risk
    z_max: float = 30.0           # distance above this gives zero distance risk
    half_width_c: float = 1.3     # lateral half-width of the collision corridor (m)
    length_c: float = 1.5         # longitudinal contact margin behind the bike (m)
    side_zone_z: float = 12.0     # only consider side-swipe threats closer than this (m)
    warmup_updates: int = 8       # world-filter updates before a risk is issued
    event_sustain: int = 6        # consecutive updates risk must stay >= event_min_peak to count
    lat_close_min: float = 0.8    # min lateral closing speed (m/s) to flag a side threat
    max_rel_speed: float = 35.0   # clip implausible relative velocities (m/s)
    risk_ema: float = 0.4         # smoothing of risk score (1 = no smoothing)

    # thresholds on risk score 0-100
    th_caution: float = 25.0
    th_warning: float = 50.0
    th_danger: float = 75.0
    event_min_peak: float = 50.0  # peak risk needed to log a near-miss event

    # real-world object heights (m) used for monocular distance: Z = f * H / h_pixels
    real_height: dict = field(default_factory=lambda: {
        "car": 1.5, "truck": 3.0, "bus": 3.0, "motorcycle": 1.4,
        "bicycle": 1.5, "person": 1.7,
    })
    real_width: dict = field(default_factory=lambda: {
        "car": 1.8, "truck": 2.5, "bus": 2.5, "motorcycle": 0.8,
        "bicycle": 0.6, "person": 0.6,
    })
