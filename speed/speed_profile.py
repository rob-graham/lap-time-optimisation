"""Utilities for generating a motorcycle speed profile along a racing line.

The script reads a CSV file containing ``x,y`` coordinates that describe the
centre line of a race track.  The path is resampled to roughly equal spacing
and a simple physics model is applied to estimate the maximum speed the bike
can maintain at each point.  The resulting profile including gear selection and
engine RPM is written to a new CSV file.

Corner sequences do not require extra points to enforce tangency – the
resampler uses the heading of the preceding track point so the first arc of a
run of corners joins the incoming straight smoothly.

Run ``python speed_profile.py --help`` for a list of available command line
options.
"""

import argparse
import csv
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional
import math
import numpy as np


@dataclass
class TrackPoint:
    """A single sample on the racing line.

    ``section`` describes the upcoming segment from this point onwards and is
    either ``"straight"`` or ``"corner"``.  ``camber`` is the banking angle of
    the surface (positive slopes down to the rider's right) and ``grade`` is
    the longitudinal slope (positive when climbing).  All angles are in
    radians.
    """

    x: float
    y: float
    section: str
    camber: float
    grade: float
    radius_m: float = 0.0
    width_m: float = 0.0


def load_csv(path: str) -> List[TrackPoint]:
    """Load track points from a CSV file.

    The expected columns with a header row are ``x_m``, ``y_m``,
    ``section_type`` (``straight`` or ``corner``), ``camber_rad``,
    ``grade_rad`` and optionally ``radius_m`` (signed, metres) and ``width_m``
    (metres).
    ``section_type`` defaults to ``"corner"`` when missing or empty.
    """

    pts: List[TrackPoint] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            section = row.get("section_type", "").strip().lower()
            if not section:
                section = "corner"
            pts.append(
                TrackPoint(
                    float(row["x_m"]),
                    float(row["y_m"]),
                    section,
                    float(row.get("camber_rad", 0.0)),
                    float(row.get("grade_rad", 0.0)),
                    float(row.get("radius_m", 0.0)),
                    float(row.get("width_m", 0.0)),
                )
            )
    return pts


def cumulative_distance(pts: List[TrackPoint]) -> List[float]:
    """Return cumulative distance along *pts* starting from zero."""
    s = [0.0]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1].x, pts[i - 1].y
        x1, y1 = pts[i].x, pts[i].y
        s.append(s[-1] + math.hypot(x1 - x0, y1 - y0))
    return s


def save_csv(
    path: str,
    pts: List[TrackPoint],
    dists: List[float],
    speeds: List[float],
    gears: List[int],
    rpms: List[float],
    curvatures: List[float],
    limiters: List[str],
) -> None:
    """Save results including curvature, section type and limiting factor."""
    n = len(pts)
    if not (
        n
        == len(dists)
        == len(speeds)
        == len(gears)
        == len(rpms)
        == len(curvatures)
        == len(limiters)
    ):
        raise ValueError("all input lists must have the same length")
    
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "x_m",
                "y_m",
                "s_m",
                "v_mps",
                "v_kph",
                "gear",
                "rpm",
                "curvature_1pm",
                "section_type",
                "camber_rad",
                "grade_rad",
                "limit",
            ]
        )
        for pt, s, v, g, r, curv, lim in zip(
            pts, dists, speeds, gears, rpms, curvatures, limiters
        ):
            writer.writerow(
                [
                    pt.x,
                    pt.y,
                    s,
                    v,
                    v * 3.6,
                    g,
                    r,
                    curv,
                    pt.section,
                    pt.camber,
                    pt.grade,
                    lim,
                ]
            )


def _arc_segment(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    step: float,
) -> List[Tuple[float, float]]:
    """Return points from *p1* to *p2* following a circular arc through
    (*p0*, *p1*, *p2*).  If the points are nearly collinear a straight line is
    generated instead."""
    x1, y1 = p0
    x2, y2 = p1
    x3, y3 = p2
    area = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
    if abs(area) < 1e-9:
        seg_len = math.hypot(x3 - x2, y3 - y2)
        n = max(1, int(math.floor(seg_len / step)))
        return [
            (x2 + (j / n) * (x3 - x2), y2 + (j / n) * (y3 - y2))
            for j in range(n + 1)
        ]

    b = (x1 * x1 + y1 * y1) * (y3 - y2) + (x2 * x2 + y2 * y2) * (y1 - y3) + (
        x3 * x3 + y3 * y3
    ) * (y2 - y1)
    c = (x1 * x1 + y1 * y1) * (x2 - x3) + (x2 * x2 + y2 * y2) * (x3 - x1) + (
        x3 * x3 + y3 * y3
    ) * (x1 - x2)
    cx = -b / (2 * area)
    cy = -c / (2 * area)
    r = math.hypot(x2 - cx, y2 - cy)
    th1 = math.atan2(y2 - cy, x2 - cx)
    th2 = math.atan2(y3 - cy, x3 - cx)
    cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
    if cross > 0.0:
        if th2 <= th1:
            th2 += 2 * math.pi
    else:
        if th2 >= th1:
            th2 -= 2 * math.pi
    arc_len = r * abs(th2 - th1)
    n = max(1, int(math.floor(arc_len / step)))
    return [
        (cx + r * math.cos(th1 + (j / n) * (th2 - th1)),
         cy + r * math.sin(th1 + (j / n) * (th2 - th1)))
        for j in range(n + 1)
    ]


def _arc_from_radius(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    radius: float,
    step: float,
    heading: float | None = None,
) -> List[Tuple[float, float]]:
    """Return points along a circular arc from *p0* to *p1* with signed
    *radius*.

    When *heading* is provided it specifies the incoming tangent direction at
    ``p0`` (in radians from the +x axis) so the arc begins tangent to that
    heading.  Otherwise the arc is defined solely by the chord between ``p0``
    and ``p1``.
    """

    if radius == 0.0 or math.isinf(radius):
        return _arc_segment(p0, p0, p1, step)

    x0, y0 = p0
    x1, y1 = p1

    if heading is not None:
        sign = 1.0 if radius >= 0 else -1.0
        cx = x0 - math.sin(heading) * radius
        cy = y0 + math.cos(heading) * radius
        r = abs(radius)
        th0 = math.atan2(y0 - cy, x0 - cx)
        th1 = math.atan2(y1 - cy, x1 - cx)
        if sign > 0 and th1 <= th0:
            th1 += 2 * math.pi
        elif sign < 0 and th1 >= th0:
            th1 -= 2 * math.pi
        arc_len = r * abs(th1 - th0)
        n = max(1, int(math.floor(arc_len / step)))
        return [
            (cx + r * math.cos(th0 + (j / n) * (th1 - th0)),
             cy + r * math.sin(th0 + (j / n) * (th1 - th0)))
            for j in range(n + 1)
        ]

    dx = x1 - x0
    dy = y1 - y0
    chord = math.hypot(dx, dy)
    if chord == 0.0:
        return [(x0, y0)]

    r = abs(radius)
    # Clamp to valid geometry.  If the requested radius is too small for the
    # chord length, fall back to the minimum possible radius.
    if r < chord / 2:
        r = chord / 2

    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    h = math.sqrt(max(r * r - (chord / 2.0) ** 2, 0.0))
    ux, uy = dx / chord, dy / chord
    px, py = -uy, ux
    sign = 1.0 if radius >= 0 else -1.0
    cx = mx + sign * h * px
    cy = my + sign * h * py

    th0 = math.atan2(y0 - cy, x0 - cx)
    th1 = math.atan2(y1 - cy, x1 - cx)
    if sign > 0 and th1 <= th0:
        th1 += 2 * math.pi
    elif sign < 0 and th1 >= th0:
        th1 -= 2 * math.pi

    arc_len = r * abs(th1 - th0)
    n = max(1, int(math.floor(arc_len / step)))
    return [
        (cx + r * math.cos(th0 + (j / n) * (th1 - th0)),
         cy + r * math.sin(th0 + (j / n) * (th1 - th0)))
        for j in range(n + 1)
    ]


def resample(points: List[TrackPoint], step: float, closed: bool = True) -> List[TrackPoint]:
    """Resample *points* so adjacent samples are roughly ``step`` metres apart.

    When ``closed`` is ``True`` the path is treated as a closed loop and the
    first point is appended to the end if necessary. For open tracks the
    initial heading is derived from the first segment and the duplicate end
    point is omitted.
    """

    if len(points) < 2:
        return points

    if closed and math.hypot(points[0].x - points[-1].x, points[0].y - points[-1].y) > 1e-6:
        points = points + [points[0]]

    resampled: List[TrackPoint] = []
    if closed:
        heading = math.atan2(points[0].y - points[-2].y, points[0].x - points[-2].x)
    else:
        heading = math.atan2(points[1].y - points[0].y, points[1].x - points[0].x)

    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        if start.section == "corner":
            seg = _arc_from_radius((start.x, start.y), (end.x, end.y), start.radius_m, step, heading)
        else:
            seg = _arc_segment((start.x, start.y), (start.x, start.y), (end.x, end.y), step)
        pts_iter = seg if i == 0 else seg[1:]
        for x, y in pts_iter:
            resampled.append(TrackPoint(x, y, start.section, start.camber, start.grade, start.radius_m, start.width_m))
        if len(seg) >= 2:
            heading = math.atan2(seg[-1][1] - seg[-2][1], seg[-1][0] - seg[-2][0])

    if closed and resampled and (
        resampled[0].x != resampled[-1].x or resampled[0].y != resampled[-1].y
    ):
        resampled.append(resampled[0])

    return resampled


def _curvature(p0: TrackPoint, p1: TrackPoint, p2: TrackPoint) -> float:
    """Return curvature (1/radius) for three consecutive points."""
    x1, y1 = p0.x, p0.y
    x2, y2 = p1.x, p1.y
    x3, y3 = p2.x, p2.y
    a = math.hypot(x2 - x1, y2 - y1)
    b = math.hypot(x3 - x2, y3 - y2)
    c = math.hypot(x3 - x1, y3 - y1)
    area = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / 2.0
    if area == 0:
        return 0.0
    radius = a * b * c / (4.0 * area)
    return 1.0 / radius


@dataclass
class BikeParams:
    """Parameters describing the motorcycle and environment.

    The defaults approximate a 600-class sport bike.  Adjust these values to
    model different machines or track conditions.  All forces are expressed in
    SI units.
    """

    rho: float = 1.225
    g: float = 9.81
    m: float = 265.0
    CdA: float = 0.35
    Crr: float = 0.015
    rw: float = 0.31
    mu: float = 1.20
    a_wheelie_max: float = 1.0 * 9.81   # was 0.9 * 9.81
    a_brake: float = 1.2 * 9.81
    shift_rpm: float = 16000.0
    primary: float = 85.0/41.0
    final_drive: float = 47.0/16.0
    gears: Tuple[float, ...] = (2.583, 2.000, 1.667, 1.444, 1.286, 1.150)
    eta_driveline: float = 0.95
    T_peak: float = 63.0
    phi_max_deg: Optional[float] = None
    kappa_dot_max: Optional[float] = None
    use_lean_angle_cap: bool = False
    use_steer_rate_cap: bool = False

    def torque_curve(self, rpm: float) -> float:
        """Simple flat torque curve."""
        return self.T_peak


def engine_rpm(v: float, bp: BikeParams, gear: float) -> float:
    omega_w = v / bp.rw
    omega_e = omega_w * bp.primary * bp.final_drive * gear
    return omega_e * 60.0 / (2 * math.pi)


def wheel_force(v: float, bp: BikeParams, gear: float) -> float:
    T = bp.torque_curve(engine_rpm(v, bp, gear))
    G = bp.primary * bp.final_drive * gear
    return (T * G * bp.eta_driveline) / bp.rw


def aero_drag(v: float, bp: BikeParams) -> float:
    return 0.5 * bp.rho * bp.CdA * v * v


def roll_res(bp: BikeParams) -> float:
    return bp.Crr * bp.m * bp.g


def traction_circle_cap(
    v: float,
    radius: float,
    bp: BikeParams,
    camber: float,
    grade: float,
    eps: float = 0.0,
) -> float:
    """Return longitudinal acceleration limit from the traction circle."""

    radius = max(abs(radius), 1.0)
    # Lateral demand is reduced by track camber.
    a_lat = (v * v) / radius - bp.g * math.sin(camber)
    a_max = bp.mu * bp.g * math.cos(grade) * math.cos(camber)
    inside = (a_max * a_max) * (1.0 - eps) - a_lat * a_lat
    return math.sqrt(max(0.0, inside))


def select_gear(v: float, bp: BikeParams) -> float:
    """Return the lowest gear whose engine speed does not exceed ``shift_rpm``."""
    # ``bp.gears`` is expected to be ordered from first to top gear
    for g in bp.gears:
        if engine_rpm(v, bp, g) <= bp.shift_rpm:
            return g
    # fall back to top gear if all would exceed the shift point
    return bp.gears[-1]


def max_speed(bp: BikeParams) -> float:
    """Maximum speed attainable at the shift RPM in top gear."""
    top = bp.gears[-1]
    omega_e = bp.shift_rpm * 2 * math.pi / 60.0
    omega_w = omega_e / (bp.primary * bp.final_drive * top)
    return omega_w * bp.rw


def compute_speed_profile(
    pts: List[TrackPoint],
    bp: BikeParams,
    use_traction_circle: bool = False,
    trail_braking: bool = False,
    sweeps: int = 8,
    curv_smoothing: int = 3,
    speed_smoothing: int = 3,
    phi_max_deg: Optional[float] = None,
    kappa_dot_max: Optional[float] = None,
    use_lean_angle_cap: bool = False,
    use_steer_rate_cap: bool = False,
    closed_loop: bool = True,
) -> Tuple[List[float], float, List[float], List[str]]:
    """Compute a speed profile, curvature and limiting factor for *pts*.

    The solver performs a number of forward and backward sweeps along the
    resampled path, applying acceleration limits from the engine, aerodynamics
    and optional traction circle.  When ``trail_braking`` is true the same
    traction limit is applied while decelerating.  ``curv_smoothing`` and
    ``speed_smoothing`` control neighbour-averaging passes for corner radius and
    final speeds respectively, which can help remove jitter on high resolution
    tracks.  Additional optional limits can be applied via ``phi_max_deg`` and
    ``kappa_dot_max`` to cap lean angle and steer rate.
    The function returns a list of speeds in metres per second for each path
    point and the overall lap time in seconds.
    """

    n = len(pts)
    if n < 3:
        return [0.0] * n, 0.0

    x = [p.x for p in pts]
    y = [p.y for p in pts]
    camber = [p.camber for p in pts]
    grade = [p.grade for p in pts]
    section = [p.section for p in pts]
    ds = [math.hypot(x[i + 1] - x[i], y[i + 1] - y[i]) for i in range(n - 1)]
    s_vals = [0.0]
    for d in ds:
        s_vals.append(s_vals[-1] + d)
    s = np.array(s_vals)

    R = [math.inf] * n
    for i in range(n):
        if section[i] == "corner":
            if pts[i].radius_m != 0.0:
                R[i] = abs(pts[i].radius_m)
            elif 0 < i < n - 1:
                curv = _curvature(pts[i - 1], pts[i], pts[i + 1])
                R[i] = 1.0 / max(abs(curv), 1e-9)
    if section[0] == "corner" and not math.isfinite(R[0]):
        R[0] = R[1]
    if section[-1] == "corner" and not math.isfinite(R[-1]):
        R[-1] = R[-2]

    for _ in range(curv_smoothing):
        R_s = R.copy()
        if closed_loop:
            for i in range(n):
                if section[i] == "corner":
                    im1 = (i - 1) % n
                    ip1 = (i + 1) % n
                    if section[im1] == "corner" and section[ip1] == "corner":
                        R_s[i] = 0.25 * R[im1] + 0.5 * R[i] + 0.25 * R[ip1]
        else:
            for i in range(1, n - 1):
                if section[i] == "corner" and section[i - 1] == "corner" and section[i + 1] == "corner":
                    R_s[i] = 0.25 * R[i - 1] + 0.5 * R[i] + 0.25 * R[i + 1]
        R = R_s

    kappa = np.zeros(n)
    for i in range(n):
        if math.isfinite(R[i]):
            kappa[i] = 1.0 / max(R[i], 1e-9)

    if phi_max_deg is not None and use_lean_angle_cap:
        phi_max_rad = math.radians(phi_max_deg)
        v_lean = np.sqrt(bp.g * math.tan(phi_max_rad) / np.maximum(kappa, 1e-6))
    else:
        v_lean = np.full(n, math.inf)

    if kappa_dot_max is not None and use_steer_rate_cap:
        dkappa_ds = np.gradient(kappa, s, edge_order=2)
        window = 5 if n >= 5 else 3 if n >= 3 else 1
        if window > 1:
            kernel = np.ones(window) / window
            dkappa_ds = np.convolve(dkappa_ds, kernel, mode="same")
        v_steer = kappa_dot_max / np.maximum(np.abs(dkappa_ds), 1e-6)
    else:
        v_steer = np.full(n, math.inf)

    v: List[float] = []
    limit_reason: List[str] = []
    v_top = max_speed(bp)
    for i in range(n):
        if section[i] == "corner":
            a_lat_cap = bp.mu * bp.g * math.cos(grade[i]) * math.cos(camber[i])
            v0 = math.sqrt(
                max(0.0, R[i] * (a_lat_cap + bp.g * math.sin(camber[i])))
            )
            limiter = "corner"
            if use_lean_angle_cap and phi_max_deg is not None and v0 > v_lean[i]:
                v0 = v_lean[i]
                limiter = "lean"
            if use_steer_rate_cap and kappa_dot_max is not None and v0 > v_steer[i]:
                v0 = v_steer[i]
                limiter = "steer"
            if v0 > v_top:
                v0 = v_top
                limiter = "rpm"
            v.append(v0)
            limit_reason.append(limiter)
        else:
            v0 = 100.0
            limiter = "power"
            if use_lean_angle_cap and phi_max_deg is not None and v0 > v_lean[i]:
                v0 = v_lean[i]
                limiter = "lean"
            if use_steer_rate_cap and kappa_dot_max is not None and v0 > v_steer[i]:
                v0 = v_steer[i]
                limiter = "steer"
            if v0 > v_top:
                v0 = v_top
                limiter = "rpm"
            v.append(v0)
            limit_reason.append(limiter)

    for _ in range(sweeps):
        for i in range(n - 1):
            v_i = v[i]
            gear = select_gear(v_i, bp)
            Fw = wheel_force(v_i, bp, gear)
            a_base = (Fw - aero_drag(v_i, bp) - roll_res(bp)) / bp.m
            a = a_base - bp.g * math.sin(grade[i])
            limiter = "accel" if a >= 0 else "braking"
            if a >= 0 and a > bp.a_wheelie_max:
                a = bp.a_wheelie_max
                limiter = "wheelie"
            if a < 0 and -a > bp.a_brake:
                a = -bp.a_brake
                limiter = "stoppie"
            if use_traction_circle:
                cap = traction_circle_cap(v_i, R[i], bp, camber[i], grade[i])
                if abs(a) > cap:
                    a = cap if a >= 0 else -cap
                    limiter = "corner"
            v_next = math.sqrt(max(0.0, v_i * v_i + 2 * a * ds[i]))
            if v_next > v_top:
                v_next = v_top
                limiter = "rpm"
            if use_lean_angle_cap and phi_max_deg is not None and v_next > v_lean[i + 1]:
                v_next = v_lean[i + 1]
                limiter = "lean"
            if use_steer_rate_cap and kappa_dot_max is not None and v_next > v_steer[i + 1]:
                v_next = v_steer[i + 1]
                limiter = "steer"
            if v_next < v[i + 1]:
                v[i + 1] = v_next
                limit_reason[i + 1] = limiter

        for i in range(n - 2, -1, -1):
            a_brk = bp.a_brake
            limiter = "stoppie"
            if use_traction_circle and trail_braking:
                cap = traction_circle_cap(v[i + 1], R[i + 1], bp, camber[i + 1], grade[i + 1])
                if cap < a_brk:
                    a_brk = cap
                    limiter = "corner"
            a_tot = a_brk + bp.g * math.sin(grade[i])
            v_prev = math.sqrt(max(0.0, v[i + 1] * v[i + 1] + 2 * a_tot * ds[i]))
            if v_prev > v_top:
                v_prev = v_top
                limiter = "rpm"
            if use_lean_angle_cap and phi_max_deg is not None and v_prev > v_lean[i]:
                v_prev = v_lean[i]
                limiter = "lean"
            if use_steer_rate_cap and kappa_dot_max is not None and v_prev > v_steer[i]:
                v_prev = v_steer[i]
                limiter = "steer"
            if v_prev < v[i]:
                v[i] = v_prev
                limit_reason[i] = "braking" if limiter == "stoppie" else limiter
        if closed_loop:
            v_loop = min(v[0], v[-1])
            v[0] = v[-1] = v_loop

    # simple neighbour averaging to damp residual jitter
    for _ in range(speed_smoothing):
        v_smooth = v.copy()
        if closed_loop:
            for i in range(n):
                v_smooth[i] = 0.25 * v[(i - 1) % n] + 0.5 * v[i] + 0.25 * v[(i + 1) % n]
                if v_smooth[i] > v_top:
                    v_smooth[i] = v_top
                    limit_reason[i] = "rpm"
            v = v_smooth
            v_loop = min(v[0], v[-1])
            v[0] = v[-1] = v_loop
        else:
            for i in range(n):
                if i == 0:
                    v_smooth[i] = 0.5 * v[i] + 0.5 * v[i + 1]
                elif i == n - 1:
                    v_smooth[i] = 0.5 * v[i] + 0.5 * v[i - 1]
                else:
                    v_smooth[i] = 0.25 * v[i - 1] + 0.5 * v[i] + 0.25 * v[i + 1]
                if v_smooth[i] > v_top:
                    v_smooth[i] = v_top
                    limit_reason[i] = "rpm"
            v = v_smooth

    # recompute limiting factors for deceleration segments
    for i in range(1, n):
        ds_i = ds[i - 1]
        if ds_i <= 0:
            continue
        a_seg = (v[i] * v[i] - v[i - 1] * v[i - 1]) / (2 * ds_i)
        if a_seg < 0:
            dec = -a_seg
            if dec >= bp.a_brake - 1e-6:
                limit_reason[i] = "stoppie"
            elif use_traction_circle and dec >= traction_circle_cap(v[i], R[i], bp, camber[i], grade[i]) - 1e-6:
                limit_reason[i] = "corner"
            else:
                limit_reason[i] = "braking"

    if use_lean_angle_cap and phi_max_deg is not None:
        for i in range(n):
            if math.isclose(v[i], v_lean[i], rel_tol=1e-3, abs_tol=1e-2):
                limit_reason[i] = "lean"
    if use_steer_rate_cap and kappa_dot_max is not None:
        for i in range(n):
            if math.isclose(v[i], v_steer[i], rel_tol=1e-3, abs_tol=1e-2):
                limit_reason[i] = "steer"

    lap_time = 0.0
    for i in range(n - 1):
        v_avg = max(1e-3, 0.5 * (v[i] + v[i + 1]))
        lap_time += ds[i] / v_avg

    curvatures = kappa.tolist()

    return v, lap_time, curvatures, limit_reason


@dataclass
class Segment:
    """Base class for track segments."""
    def points(self, x: float, y: float, heading: float, step: float) -> Tuple[List[Tuple[float, float]], float, float, float]:
        raise NotImplementedError


@dataclass
class Straight(Segment):
    length: float
    def points(self, x: float, y: float, heading: float, step: float):
        pts: List[Tuple[float, float]] = []
        n = max(1, int(math.ceil(self.length / step)))
        ds = self.length / n
        for _ in range(n):
            x += ds * math.cos(heading)
            y += ds * math.sin(heading)
            pts.append((x, y))
        return pts, x, y, heading


@dataclass
class Arc(Segment):
    radius: float
    angle: float  # radians, positive for left turn, negative for right
    def points(self, x: float, y: float, heading: float, step: float):
        pts: List[Tuple[float, float]] = []
        arc_len = abs(self.radius * self.angle)
        n = max(1, int(math.ceil(arc_len / step)))
        ds = arc_len / n
        sign = 1.0 if self.angle >= 0 else -1.0
        for _ in range(n):
            heading += sign * ds / self.radius
            x += ds * math.cos(heading)
            y += ds * math.sin(heading)
            pts.append((x, y))
        return pts, x, y, heading


def build_track(segments: List[Segment], step: float = 1.0,
                 start: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[Tuple[float, float]]:
    """Generate track points from straight and arc segments."""
    x, y, heading = start
    pts = [(x, y)]
    for seg in segments:
        new_pts, x, y, heading = seg.points(x, y, heading, step)
        pts.extend(new_pts)
    return pts


def _parse_gears(s: str) -> Tuple[float, ...]:
    return tuple(float(g.strip()) for g in s.split(",") if g.strip())


def main():
    parser = argparse.ArgumentParser(description="Generate speed profile for a track")
    parser.add_argument("input", help="Input track CSV")
    parser.add_argument("output", help="Output CSV with results")
    parser.add_argument("--step", type=float, default=2.0, help="Resampling distance in metres")
    parser.add_argument("--params-file", help="CSV with motorcycle parameters", default=None)

    # Motorcycle and environment parameters (override file values if provided)
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--g", type=float, default=None)
    parser.add_argument("--m", type=float, default=None)
    parser.add_argument("--CdA", type=float, default=None)
    parser.add_argument("--Crr", type=float, default=None)
    parser.add_argument("--rw", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--a_wheelie_max", type=float, default=None)
    parser.add_argument("--a_brake", type=float, default=None)
    parser.add_argument("--shift_rpm", type=float, default=None)
    parser.add_argument("--primary", type=float, default=None)
    parser.add_argument("--final_drive", type=float, default=None)
    parser.add_argument("--gears", type=str, default=None, help="Comma separated gear ratios")
    parser.add_argument("--eta_driveline", type=float, default=None)
    parser.add_argument("--T_peak", type=float, default=None)
    parser.add_argument(
        "--phi-max-deg",
        dest="phi_max_deg",
        type=float,
        default=None,
        help="Maximum lean angle in degrees",
    )
    parser.add_argument(
        "--kappa-dot-max",
        dest="kappa_dot_max",
        type=float,
        default=None,
        help="Maximum steer rate in 1/s",
    )
    parser.add_argument(
        "--use-lean-angle-cap",
        dest="use_lean_angle_cap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable lean angle speed cap",
    )
    parser.add_argument(
        "--use-steer-rate-cap",
        dest="use_steer_rate_cap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable steer rate speed cap",
    )

    parser.add_argument("--traction-circle", action="store_true",
                        help="Enable traction circle for acceleration")
    parser.add_argument("--trail-braking", action="store_true",
                        help="Apply traction circle during braking")
    parser.add_argument("--sweeps", type=int, default=25,
                        help="Forward/back sweeps for solver")
    parser.add_argument("--curv-smooth", type=int, default=3,
                        help="Neighbour-averaging passes for corner radius")
    parser.add_argument("--speed-smooth", type=int, default=3,
                        help="Neighbour-averaging passes for final speeds")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--open",
        dest="closed",
        action="store_const",
        const=False,
        help="Force track to be treated as open",
    )
    group.add_argument(
        "--closed",
        dest="closed",
        action="store_const",
        const=True,
        help="Force track to be treated as closed",
    )
    parser.set_defaults(closed=None)

    args = parser.parse_args()

    pts = load_csv(args.input)
    if args.closed is None:
        start = np.array([pts[0].x, pts[0].y])
        end = np.array([pts[-1].x, pts[-1].y])
        closed = np.allclose(start, end, atol=1e-6)
    else:
        closed = args.closed
    pts = resample(pts, args.step, closed=closed)
    dists = cumulative_distance(pts)

    bp = BikeParams()

    if args.params_file:
        gear_rows = {}
        with open(args.params_file, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                key, value = row[0], row[1]
                if key == "gears":
                    bp.gears = _parse_gears(value)
                elif key.startswith("gear") and key[4:].isdigit():
                    gear_rows[int(key[4:])] = float(value)
                elif key in {"use_lean_angle_cap", "use_steer_rate_cap"}:
                    bp.__setattr__(key, value.strip().lower() == "true")
                elif hasattr(bp, key):
                    setattr(bp, key, float(value))
        if gear_rows:
            bp.gears = tuple(gear_rows[i] for i in sorted(gear_rows))

    def _override(name: str, value):
        if value is not None:
            setattr(bp, name, value)

    _override("rho", args.rho)
    _override("g", args.g)
    _override("m", args.m)
    _override("CdA", args.CdA)
    _override("Crr", args.Crr)
    _override("rw", args.rw)
    _override("mu", args.mu)
    _override("a_wheelie_max", args.a_wheelie_max)
    _override("a_brake", args.a_brake)
    _override("shift_rpm", args.shift_rpm)
    _override("primary", args.primary)
    _override("final_drive", args.final_drive)
    if args.gears is not None:
        bp.gears = _parse_gears(args.gears)
    _override("eta_driveline", args.eta_driveline)
    _override("T_peak", args.T_peak)
    _override("phi_max_deg", args.phi_max_deg)
    _override("kappa_dot_max", args.kappa_dot_max)
    _override("use_lean_angle_cap", args.use_lean_angle_cap)
    _override("use_steer_rate_cap", args.use_steer_rate_cap)

    speeds, lap_time, curvatures, limiters = compute_speed_profile(
        pts,
        bp,
        use_traction_circle=args.traction_circle,
        trail_braking=args.trail_braking,
        sweeps=args.sweeps,
        curv_smoothing=args.curv_smooth,
        speed_smoothing=args.speed_smooth,
        phi_max_deg=bp.phi_max_deg,
        kappa_dot_max=bp.kappa_dot_max,
        use_lean_angle_cap=bp.use_lean_angle_cap,
        use_steer_rate_cap=bp.use_steer_rate_cap,
        closed_loop=closed,
    )
    gears: List[int] = []
    rpms: List[float] = []
    for v in speeds:
        gear_ratio = select_gear(v, bp)
        gears.append(bp.gears.index(gear_ratio) + 1)
        rpms.append(engine_rpm(v, bp, gear_ratio))

    save_csv(args.output, pts, dists, speeds, gears, rpms, curvatures, limiters)
    print(f"Lap time: {lap_time:.2f} s")

if __name__ == "__main__":
    main()
