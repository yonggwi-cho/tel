"""
Analyzes a single golf swing segment.

For each swing the analyzer produces:
  - Phase boundaries  (address → backswing → top → downswing → impact → follow-through)
  - Timing metrics    (phase durations in seconds, backswing/downswing ratio)
  - Rotation metrics  (shoulder and hip rotation angles at key frames)
  - Stability metrics (head displacement, spine-angle consistency)
  - Speed metric      (peak wrist speed as a proxy for club-head speed)
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter1d

from pose_extractor import FramePose, LM, get_wrist_positions, get_landmark_positions
from swing_segmenter import SwingSegment


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class PhaseTimings:
    address_to_top_sec: float        # backswing duration
    top_to_impact_sec: float         # downswing duration
    impact_to_finish_sec: float      # follow-through duration
    backswing_downswing_ratio: float # > 2 is typical for good tempo


@dataclass
class RotationMetrics:
    shoulder_rotation_deg: float   # max shoulder turn (address → top)
    hip_rotation_deg: float        # max hip turn (address → top)
    x_factor_deg: float            # shoulder − hip separation at top (ideal ≈ 45°)


@dataclass
class StabilityMetrics:
    head_displacement_px: float    # head travel in normalised coords (smaller is better)
    spine_angle_std_deg: float     # std-dev of spine tilt during swing


@dataclass
class SwingAnalysis:
    swing_id: int
    start_frame: int
    end_frame: int
    impact_frame: int
    peak_wrist_speed: float        # screen-fraction / second

    top_frame: int                 # frame index of backswing top

    timings: PhaseTimings
    rotation: RotationMetrics
    stability: StabilityMetrics

    feedback: list[str] = field(default_factory=list)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (degrees) between two 2-D vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _rotation_angle_2d(p_left: np.ndarray, p_right: np.ndarray) -> float:
    """
    Rotation angle of the line (left→right) relative to the horizontal axis.
    Positive = right side higher (typical backswing shoulder turn viewed from front).
    """
    vec = p_right - p_left
    return float(np.degrees(np.arctan2(-vec[1], vec[0])))  # y flipped in image coords


def _spine_angle(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """Angle of the torso from vertical (degrees)."""
    vec = hip_mid - shoulder_mid
    return float(np.degrees(np.arctan2(vec[0], vec[1])))


# ── Sub-frame helpers ────────────────────────────────────────────────────────

def _find_top_of_backswing(
    segment_poses: list[FramePose],
    impact_local: int,
    fps: float,
    hand: str,
) -> int:
    """
    Within the frames BEFORE impact, find the top of backswing.
    The top is where the lead wrist is at its highest (lowest y) and
    the wrist speed is locally at a minimum.
    """
    positions = get_wrist_positions(segment_poses, hand=hand)
    before_impact = positions[:impact_local, :]

    if len(before_impact) == 0 or np.all(np.isnan(before_impact[:, 1])):
        return 0

    # Smooth y-coordinate of wrist; backswing top = minimum y (highest in frame)
    y_vals = before_impact[:, 1].copy()
    nan_mask = np.isnan(y_vals)
    if nan_mask.all():
        return 0
    y_vals[nan_mask] = np.interp(
        np.where(nan_mask)[0],
        np.where(~nan_mask)[0],
        y_vals[~nan_mask],
    )
    smooth_y = gaussian_filter1d(y_vals, sigma=max(2.0, fps / 10))
    return int(np.argmin(smooth_y))


# ── Main analysis function ───────────────────────────────────────────────────

def analyze_swing(
    swing: SwingSegment,
    all_poses: list[FramePose],
    fps: float,
    hand: str = "right",
) -> SwingAnalysis:
    """
    Compute metrics for a single swing segment.

    Parameters
    ----------
    swing      : SwingSegment from detect_swings()
    all_poses  : full list of FramePose for the video
    fps        : frames per second
    hand       : "right" for a right-handed golfer (lead wrist = right wrist)
    """
    # Slice to just this swing's frames
    # Map absolute frame indices to positions in all_poses list
    seg_poses = [p for p in all_poses if swing.start_frame <= p.frame_idx <= swing.end_frame]
    if not seg_poses:
        raise ValueError(f"No poses found for swing {swing.swing_id}")

    frame_indices = [p.frame_idx for p in seg_poses]
    impact_local = _find_local(frame_indices, swing.impact_frame)

    top_local = _find_top_of_backswing(seg_poses, impact_local, fps, hand)

    # ── Timing ───────────────────────────────────────────────────────────────
    address_local = 0
    finish_local = len(seg_poses) - 1

    bs_frames = max(1, impact_local - top_local)      # backswing frames after top
    ds_frames = max(1, top_local - address_local)      # actually = top - address = backswing
    # Correct naming: backswing = address→top, downswing = top→impact
    backswing_frames = top_local - address_local
    downswing_frames = impact_local - top_local
    followthrough_frames = finish_local - impact_local

    backswing_sec = backswing_frames / fps
    downswing_sec = max(downswing_frames, 1) / fps
    followthrough_sec = followthrough_frames / fps
    ratio = backswing_sec / downswing_sec if downswing_sec > 0 else 0.0

    timings = PhaseTimings(
        address_to_top_sec=backswing_sec,
        top_to_impact_sec=downswing_sec,
        impact_to_finish_sec=followthrough_sec,
        backswing_downswing_ratio=ratio,
    )

    # ── Rotation ─────────────────────────────────────────────────────────────
    ls = get_landmark_positions(seg_poses, "left_shoulder")
    rs = get_landmark_positions(seg_poses, "right_shoulder")
    lh = get_landmark_positions(seg_poses, "left_hip")
    rh = get_landmark_positions(seg_poses, "right_hip")

    shoulder_addr = _rotation_angle_2d(_safe(ls, address_local), _safe(rs, address_local))
    shoulder_top  = _rotation_angle_2d(_safe(ls, top_local),     _safe(rs, top_local))
    hip_addr      = _rotation_angle_2d(_safe(lh, address_local), _safe(rh, address_local))
    hip_top       = _rotation_angle_2d(_safe(lh, top_local),     _safe(rh, top_local))

    shoulder_rot = abs(shoulder_top - shoulder_addr)
    hip_rot      = abs(hip_top - hip_addr)
    x_factor     = abs(shoulder_rot - hip_rot)

    rotation = RotationMetrics(
        shoulder_rotation_deg=shoulder_rot,
        hip_rotation_deg=hip_rot,
        x_factor_deg=x_factor,
    )

    # ── Stability ────────────────────────────────────────────────────────────
    nose = get_landmark_positions(seg_poses, "nose")
    valid_nose = nose[~np.any(np.isnan(nose), axis=1)]
    head_disp = float(np.max(np.linalg.norm(valid_nose - valid_nose[0], axis=1))) if len(valid_nose) > 1 else 0.0

    spine_angles = []
    for i in range(len(seg_poses)):
        s_mid = (_safe(ls, i) + _safe(rs, i)) / 2
        h_mid = (_safe(lh, i) + _safe(rh, i)) / 2
        if not (np.any(np.isnan(s_mid)) or np.any(np.isnan(h_mid))):
            spine_angles.append(_spine_angle(s_mid, h_mid))

    spine_std = float(np.std(spine_angles)) if spine_angles else 0.0

    stability = StabilityMetrics(
        head_displacement_px=head_disp,
        spine_angle_std_deg=spine_std,
    )

    # ── Feedback ─────────────────────────────────────────────────────────────
    feedback = _generate_feedback(timings, rotation, stability, swing.peak_speed)

    top_abs_frame = seg_poses[top_local].frame_idx if top_local < len(seg_poses) else swing.start_frame

    return SwingAnalysis(
        swing_id=swing.swing_id,
        start_frame=swing.start_frame,
        end_frame=swing.end_frame,
        impact_frame=swing.impact_frame,
        peak_wrist_speed=swing.peak_speed,
        top_frame=top_abs_frame,
        timings=timings,
        rotation=rotation,
        stability=stability,
        feedback=feedback,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_local(frame_indices: list[int], abs_frame: int) -> int:
    """Find the closest local index for an absolute frame number."""
    diffs = [abs(fi - abs_frame) for fi in frame_indices]
    return int(np.argmin(diffs))


def _safe(positions: np.ndarray, idx: int) -> np.ndarray:
    """Return position at idx, or zeros if NaN."""
    v = positions[idx] if idx < len(positions) else np.array([np.nan, np.nan])
    return v if not np.any(np.isnan(v)) else np.zeros(2)


def _generate_feedback(
    timings: PhaseTimings,
    rotation: RotationMetrics,
    stability: StabilityMetrics,
    peak_speed: float,
) -> list[str]:
    tips: list[str] = []

    # Tempo
    if timings.backswing_downswing_ratio < 1.5:
        tips.append("テンポ: バックスイングが速すぎます。3:1 を目標にしましょう。")
    elif timings.backswing_downswing_ratio > 4.5:
        tips.append("テンポ: ダウンスイングが遅すぎます。切り返しをスムーズに。")
    else:
        tips.append(f"テンポ: バック/ダウン比 {timings.backswing_downswing_ratio:.1f}:1 — 良好です。")

    # Shoulder turn
    if rotation.shoulder_rotation_deg < 60:
        tips.append(f"肩の回転: {rotation.shoulder_rotation_deg:.0f}° — もっと捻転を入れましょう（目標 ≥ 80°）。")
    else:
        tips.append(f"肩の回転: {rotation.shoulder_rotation_deg:.0f}° — 十分な捻転です。")

    # X-factor
    if rotation.x_factor_deg < 25:
        tips.append(f"X-ファクター: {rotation.x_factor_deg:.0f}° — 肩と腰の差をもっと作りましょう（目標 ≥ 35°）。")
    else:
        tips.append(f"X-ファクター: {rotation.x_factor_deg:.0f}° — 良好な捻転差です。")

    # Head stability
    if stability.head_displacement_px > 0.06:
        tips.append(f"頭の安定性: 移動量 {stability.head_displacement_px:.3f} — 頭を動かしすぎています。")
    else:
        tips.append(f"頭の安定性: 良好です（移動量 {stability.head_displacement_px:.3f}）。")

    # Spine angle
    if stability.spine_angle_std_deg > 8:
        tips.append(f"脊柱角度: ばらつき {stability.spine_angle_std_deg:.1f}° — スパイン角度を一定に保ちましょう。")
    else:
        tips.append(f"脊柱角度: 安定しています（ばらつき {stability.spine_angle_std_deg:.1f}°）。")

    return tips


# ── Pretty printer ───────────────────────────────────────────────────────────

def print_analysis(analysis: SwingAnalysis) -> None:
    print(f"\n{'='*60}")
    print(f"  スイング #{analysis.swing_id + 1}  "
          f"(フレーム {analysis.start_frame}–{analysis.end_frame})")
    print(f"{'='*60}")
    print(f"  インパクトフレーム : {analysis.impact_frame}")
    print(f"  ピーク手首速度    : {analysis.peak_wrist_speed:.4f} (正規化値)")
    print()
    print("  [タイミング]")
    print(f"    バックスイング   : {analysis.timings.address_to_top_sec:.2f} s")
    print(f"    ダウンスイング   : {analysis.timings.top_to_impact_sec:.2f} s")
    print(f"    フォロースルー   : {analysis.timings.impact_to_finish_sec:.2f} s")
    print(f"    テンポ比         : {analysis.timings.backswing_downswing_ratio:.1f}:1")
    print()
    print("  [回転]")
    print(f"    肩の回転         : {analysis.rotation.shoulder_rotation_deg:.1f}°")
    print(f"    腰の回転         : {analysis.rotation.hip_rotation_deg:.1f}°")
    print(f"    X-ファクター     : {analysis.rotation.x_factor_deg:.1f}°")
    print()
    print("  [安定性]")
    print(f"    頭の移動量       : {analysis.stability.head_displacement_px:.4f}")
    print(f"    脊柱角ばらつき   : {analysis.stability.spine_angle_std_deg:.1f}°")
    print()
    print("  [フィードバック]")
    for tip in analysis.feedback:
        print(f"    • {tip}")
