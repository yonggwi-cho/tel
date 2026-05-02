"""
Detects individual golf swings within a sequence of pose frames.

Strategy:
  1. Compute the speed of the lead wrist across frames.
  2. Smooth the speed signal with a Gaussian filter.
  3. Find local maxima (impact candidates) above a velocity threshold.
  4. Expand each impact point outward to find swing start/end
     (where wrist speed drops below a rest threshold).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from dataclasses import dataclass

from pose_extractor import FramePose, get_wrist_positions


@dataclass
class SwingSegment:
    swing_id: int
    start_frame: int   # inclusive
    end_frame: int     # inclusive
    impact_frame: int  # frame with peak wrist speed (approximate impact)
    peak_speed: float  # normalised wrist speed at impact


def _compute_speed(positions: np.ndarray, fps: float) -> np.ndarray:
    """Pixel-normalised speed (units: screen-fraction per second)."""
    speed = np.zeros(len(positions))
    for i in range(1, len(positions)):
        if not (np.any(np.isnan(positions[i])) or np.any(np.isnan(positions[i - 1]))):
            speed[i] = np.linalg.norm(positions[i] - positions[i - 1]) * fps
    return speed


def _fill_nan_linear(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate over NaN gaps."""
    out = arr.copy()
    nans = np.isnan(out)
    if nans.any():
        x = np.arange(len(out))
        ok = ~nans
        if ok.sum() >= 2:
            out[nans] = np.interp(x[nans], x[ok], out[ok])
    return out


def detect_swings(
    poses: list[FramePose],
    fps: float,
    hand: str = "right",
    smooth_sigma: float = 3.0,
    peak_threshold_quantile: float = 0.85,
    rest_threshold_quantile: float = 0.30,
    min_swing_frames: int = 15,
    min_gap_frames: int = 10,
) -> list[SwingSegment]:
    """
    Segment a pose sequence into individual golf swings.

    Parameters
    ----------
    poses : list[FramePose]
    fps   : frames per second of the original video
    hand  : which wrist to track ("right" for right-handed golfer lead wrist)
    smooth_sigma          : Gaussian smoothing kernel (frames)
    peak_threshold_quantile  : quantile of the speed distribution used as the
                               minimum height for an impact peak
    rest_threshold_quantile  : quantile below which the wrist is considered at rest
    min_swing_frames      : discard segments shorter than this
    min_gap_frames        : merge impact peaks closer than this (frames)

    Returns
    -------
    list of SwingSegment
    """
    raw_positions = get_wrist_positions(poses, hand=hand)
    positions = np.column_stack([
        _fill_nan_linear(raw_positions[:, 0]),
        _fill_nan_linear(raw_positions[:, 1]),
    ])

    speed = _compute_speed(positions, fps)
    smooth_speed = gaussian_filter1d(speed, sigma=smooth_sigma)

    nonzero = smooth_speed[smooth_speed > 0]
    if len(nonzero) == 0:
        return []

    peak_threshold = float(np.quantile(nonzero, peak_threshold_quantile))
    rest_threshold = float(np.quantile(nonzero, rest_threshold_quantile))

    peaks, props = find_peaks(
        smooth_speed,
        height=peak_threshold,
        distance=min_gap_frames,
    )

    if len(peaks) == 0:
        return []

    segments: list[SwingSegment] = []
    n = len(smooth_speed)

    for swing_id, peak in enumerate(peaks):
        # Walk backward to find start (speed drops below rest threshold)
        start = peak
        while start > 0 and smooth_speed[start] > rest_threshold:
            start -= 1

        # Walk forward to find end
        end = peak
        while end < n - 1 and smooth_speed[end] > rest_threshold:
            end += 1

        if (end - start) < min_swing_frames:
            continue

        segments.append(SwingSegment(
            swing_id=swing_id,
            start_frame=int(poses[start].frame_idx),
            end_frame=int(poses[end].frame_idx),
            impact_frame=int(poses[peak].frame_idx),
            peak_speed=float(smooth_speed[peak]),
        ))

    # Remove overlapping segments (keep the one with higher peak speed)
    segments = _remove_overlaps(segments)
    # Re-number sequentially
    for i, seg in enumerate(segments):
        seg.swing_id = i

    return segments


def _remove_overlaps(segments: list[SwingSegment]) -> list[SwingSegment]:
    if not segments:
        return segments
    segments = sorted(segments, key=lambda s: s.start_frame)
    kept = [segments[0]]
    for seg in segments[1:]:
        if seg.start_frame <= kept[-1].end_frame:
            # Overlap – keep the one with the larger peak speed
            if seg.peak_speed > kept[-1].peak_speed:
                kept[-1] = seg
        else:
            kept.append(seg)
    return kept
