"""Physical-feasibility filtering for M2T2 grasp candidates (roadmap step G2).

The current pipeline picks grasps by raw network confidence (``argmax``) with no
physical check. This module rejects candidates that are not physically graspable
on the target platform *before* ranking, and is intentionally standalone
(NumPy-only core, optional ``ur_ikfast``) so it can be unit-tested without the
full DL stack and later wrapped in a ROS 2 node.

Input grasps are 4x4 homogeneous poses already expressed in the robot base frame
(as produced by ``class_m2t2`` after the ``cam_pose`` transform). For each grasp
we check:

    1. opening width  : required width must fit the gripper stroke
    2. approach cone   : approach direction must stay within a cone (e.g. top-down)
    3. collision       : the gripper solid model must not intersect the scene cloud
    4. reachability    : an analytic UR inverse-kinematics solution must exist

Target test platform: UR5e + Robotiq Hand-E + Orbbec Gemini 2L (all configurable).

Grasp-frame convention (configurable, MUST be validated against M2T2's output):
the gripper approaches along the local +Z axis and closes along the local X axis.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

# Optional analytic IK for the UR series. Imported lazily so this module also
# works before the (heavy) environment is installed; when missing, the IK check
# is skipped instead of failing.
try:
    from ur_ikfast import ur_kinematics as _ur_kinematics
    _HAS_IKFAST = True
except Exception:  # pragma: no cover - depends on environment
    _ur_kinematics = None
    _HAS_IKFAST = False


@dataclass
class GripperParams:
    """Parallel-jaw gripper geometry. Defaults model a Robotiq Hand-E (~50 mm)."""

    max_width: float = 0.050        # m, full stroke
    min_width: float = 0.0          # m
    finger_length: float = 0.045    # m, finger extent along approach (+Z)
    finger_thickness: float = 0.014 # m, finger extent along closing axis (X)
    finger_width: float = 0.024     # m, finger extent along Y
    palm_depth: float = 0.060       # m, body extent behind the TCP (-Z)
    palm_radius: float = 0.040      # m, body half-extent in X and Y
    width_margin: float = 0.005     # m, safety added to the required opening


@dataclass
class FeasibilityConfig:
    """Tunable thresholds and conventions for the feasibility checks."""

    approach_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    closing_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    # Preferred approach direction in the robot base frame (top-down = -Z).
    preferred_approach_world: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    max_approach_tilt_deg: float = 60.0   # reject grasps tilted more than this
    collision_clearance: float = 0.005    # m, expand gripper solids by this margin
    max_collision_points: int = 0         # tolerate at most this many scene points
    default_width: float = 0.040          # used when contacts/width are unknown
    # Reachability (analytic IK).
    robot_model: str = "ur5e"
    check_ik: bool = True


# --------------------------------------------------------------------------- #
# Geometry helpers (pure NumPy)
# --------------------------------------------------------------------------- #
def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def approach_angle_deg(pose, cfg):
    """Angle (deg) between the grasp approach direction and the preferred one."""
    R = np.asarray(pose)[:3, :3]
    approach_world = _unit(R @ _unit(cfg.approach_axis))
    pref = _unit(cfg.preferred_approach_world)
    cos = float(np.clip(np.dot(approach_world, pref), -1.0, 1.0))
    return np.degrees(np.arccos(cos))


def required_width(contact_pair):
    """Opening width (m) from a pair of antipodal contact points, or None."""
    if contact_pair is None:
        return None
    c = np.asarray(contact_pair, dtype=float)
    if c.shape != (2, 3):
        return None
    return float(np.linalg.norm(c[0] - c[1]))


def _points_in_box(p_local, lo, hi):
    """Boolean mask of points inside the axis-aligned box [lo, hi] (local frame)."""
    return np.all((p_local >= lo) & (p_local <= hi), axis=1)


def gripper_collision_count(pose, scene_xyz, width, gripper, cfg):
    """Number of scene points intersecting the gripper solids at this grasp.

    The object sitting between the fingers (the "throat") is intentionally not
    counted as a collision; only the two finger solids and the palm body are.
    """
    if scene_xyz is None or len(scene_xyz) == 0:
        return 0
    pose = np.asarray(pose)
    R, t = pose[:3, :3], pose[:3, 3]
    # Transform scene points into the grasp frame: p_local = R^T (p - t).
    p_local = (np.asarray(scene_xyz, dtype=float) - t) @ R

    m = cfg.collision_clearance
    half_w = max(width, 0.0) / 2.0
    th = gripper.finger_thickness
    fy = gripper.finger_width / 2.0
    fl = gripper.finger_length

    # Right finger: x in [+half_w, +half_w+th]; Left finger: x in [-half_w-th, -half_w].
    right = _points_in_box(
        p_local, np.array([half_w - m, -fy - m, -m]), np.array([half_w + th + m, fy + m, fl + m])
    )
    left = _points_in_box(
        p_local, np.array([-half_w - th - m, -fy - m, -m]), np.array([-half_w + m, fy + m, fl + m])
    )
    # Palm body behind the TCP (negative Z).
    pr = gripper.palm_radius
    palm = _points_in_box(
        p_local, np.array([-pr - m, -pr - m, -gripper.palm_depth - m]), np.array([pr + m, pr + m, m])
    )
    return int(np.count_nonzero(right | left | palm))


# --------------------------------------------------------------------------- #
# Reachability (analytic IK, optional)
# --------------------------------------------------------------------------- #
class _IKChecker:
    """Thin wrapper around ur_ikfast; degrades to 'unknown' when unavailable."""

    def __init__(self, cfg):
        self.enabled = cfg.check_ik and _HAS_IKFAST
        self._arm = None
        if cfg.check_ik and not _HAS_IKFAST:
            warnings.warn(
                "ur_ikfast not installed: reachability (IK) check skipped. "
                "Install it to enable the UR5e IK filter.",
                RuntimeWarning,
            )
        if self.enabled:
            self._arm = _ur_kinematics.URKinematics(cfg.robot_model)

    def reachable(self, pose):
        """True/False if IK is available, else None (unknown -> treated as pass)."""
        if not self.enabled:
            return None
        try:
            R = np.asarray(pose)[:3, :3]
            t = np.asarray(pose)[:3, 3]
            quat = _matrix_to_quat_wxyz(R)
            ee = np.concatenate([t, quat])  # [x, y, z, qw, qx, qy, qz]
            sol = self._arm.inverse(ee, all_solutions=True)
            return sol is not None and len(np.atleast_2d(sol)) > 0
        except Exception:  # pragma: no cover - depends on ur_ikfast build
            return None


def _matrix_to_quat_wxyz(R):
    """Rotation matrix -> quaternion [w, x, y, z] (NumPy, no SciPy dependency)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def evaluate_grasp(pose, scene_xyz, contact_pair, gripper, cfg, ik=None):
    """Evaluate a single grasp; returns a report dict (no ranking)."""
    width = required_width(contact_pair)
    if width is None:
        width = cfg.default_width
    width_cmd = min(width + gripper.width_margin, gripper.max_width)

    angle = approach_angle_deg(pose, cfg)
    n_col = gripper_collision_count(pose, scene_xyz, width_cmd, gripper, cfg)
    ik_ok = ik.reachable(pose) if ik is not None else None

    reasons = []
    width_ok = gripper.min_width <= width <= gripper.max_width
    if not width_ok:
        reasons.append(f"width {width*1000:.1f}mm outside [{gripper.min_width*1000:.0f},"
                       f"{gripper.max_width*1000:.0f}]mm")
    approach_ok = angle <= cfg.max_approach_tilt_deg
    if not approach_ok:
        reasons.append(f"approach tilt {angle:.0f}deg > {cfg.max_approach_tilt_deg:.0f}deg")
    collision_ok = n_col <= cfg.max_collision_points
    if not collision_ok:
        reasons.append(f"{n_col} scene points collide with gripper")
    ik_ok_eff = ik_ok in (True, None)
    if ik_ok is False:
        reasons.append("no IK solution (unreachable)")

    feasible = width_ok and approach_ok and collision_ok and ik_ok_eff
    return {
        "feasible": feasible,
        "width": width,
        "approach_deg": angle,
        "collision_points": n_col,
        "ik_ok": ik_ok,
        "reasons": reasons,
    }


def filter_grasps(grasps, confidences, scene_xyz=None, contacts=None,
                  gripper=None, cfg=None):
    """Filter and confidence-rank a set of grasp candidates.

    Args:
        grasps: array-like of N 4x4 poses (robot base frame).
        confidences: array-like of N scores.
        scene_xyz: (M,3) scene point cloud for collision (optional).
        contacts: optional N x 2 x 3 antipodal contact points (for width).
        gripper: GripperParams (defaults to Hand-E).
        cfg: FeasibilityConfig.

    Returns:
        feasible_idx: indices of feasible grasps sorted by descending confidence.
        reports: list of per-grasp report dicts (same order as input).
    """
    gripper = gripper or GripperParams()
    cfg = cfg or FeasibilityConfig()
    grasps = np.asarray(grasps, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    ik = _IKChecker(cfg)

    reports = []
    for i in range(len(grasps)):
        cp = None if contacts is None else contacts[i]
        reports.append(evaluate_grasp(grasps[i], scene_xyz, cp, gripper, cfg, ik))

    feasible_idx = [i for i, r in enumerate(reports) if r["feasible"]]
    feasible_idx.sort(key=lambda i: confidences[i], reverse=True)
    return feasible_idx, reports


# --------------------------------------------------------------------------- #
# Synthetic self-test (NumPy only, no torch / no ur_ikfast required)
# --------------------------------------------------------------------------- #
def _make_topdown_pose(center, yaw=0.0):
    """Top-down grasp: local +Z -> world -Z; local X (closing) rotated by yaw in XY."""
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0, 0, 1.0]])
    flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])  # approach -> -Z
    T = np.eye(4)
    T[:3, :3] = Rz @ flip
    T[:3, 3] = center
    return T


def _self_test():
    gripper = GripperParams()
    cfg = FeasibilityConfig(check_ik=False)  # IK off for the offline test

    # Ground plane z=0 plus a small object near (0.4, 0.0, 0.05).
    gx, gy = np.meshgrid(np.linspace(0.2, 0.6, 40), np.linspace(-0.2, 0.2, 40))
    ground = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)
    obj_center = np.array([0.4, 0.0, 0.05])
    scene = np.vstack([ground, obj_center + 0.01 * (np.random.rand(50, 3) - 0.5)])

    # 1) good top-down grasp just above the object -> feasible
    good = _make_topdown_pose(obj_center + np.array([0, 0, 0.02]))
    # 2) too-wide contacts (60 mm > 50 mm stroke) -> width reject
    wide_contacts = np.array([[0.4, -0.03, 0.05], [0.4, 0.03, 0.05]])
    # 3) sideways grasp (approach horizontal) -> approach-cone reject
    sideways = np.eye(4); sideways[:3, 3] = obj_center  # approach = +Z (up), 180deg off
    # 4) grasp sunk into the ground plane -> collision reject
    sunk = _make_topdown_pose(np.array([0.4, 0.0, -0.02]))

    grasps = np.array([good, good, sideways, sunk])
    confs = np.array([0.9, 0.7, 0.99, 0.95])
    contacts = [None, wide_contacts, None, None]

    feas, reps = filter_grasps(grasps, confs, scene, contacts, gripper, cfg)
    for i, r in enumerate(reps):
        tag = "FEASIBLE" if r["feasible"] else "rejected"
        print(f"  grasp {i}: {tag:9s} angle={r['approach_deg']:5.1f}deg "
              f"col={r['collision_points']:3d} width={r['width']*1000:.0f}mm "
              f"{'| ' + '; '.join(r['reasons']) if r['reasons'] else ''}")
    print(f"  -> feasible (ranked by confidence): {feas}")

    assert reps[0]["feasible"], "good grasp should be feasible"
    assert not reps[1]["feasible"] and "width" in reps[1]["reasons"][0], "wide grasp -> width reject"
    assert not reps[2]["feasible"], "sideways grasp -> approach reject"
    assert not reps[3]["feasible"], "sunk grasp -> collision reject"
    assert feas == [0], f"only grasp 0 feasible, got {feas}"
    print("Self-test PASSED ✓")


if __name__ == "__main__":
    _self_test()
