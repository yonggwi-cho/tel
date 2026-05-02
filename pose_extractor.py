"""
Extracts MediaPipe pose landmarks from every frame of a video.
Returns a list of per-frame landmark arrays (33 landmarks × [x, y, z, visibility]).
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional


# MediaPipe landmark indices used by the rest of the pipeline
LM = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class FramePose:
    frame_idx: int
    timestamp_ms: float
    landmarks: Optional[np.ndarray]  # (33, 4): x, y, z, visibility; None if not detected


def extract_poses(video_path: str, skip_frames: int = 0) -> tuple[list[FramePose], dict]:
    """
    Run MediaPipe Pose on every (1 + skip_frames)-th frame.

    Returns:
        poses: list of FramePose for each processed frame
        video_info: dict with fps, width, height, total_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_info = {"fps": fps, "width": width, "height": height, "total_frames": total_frames}

    mp_pose = mp.solutions.pose
    poses: list[FramePose] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose_model:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose_model.process(rgb)

            if result.pose_landmarks:
                lm_array = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
            else:
                lm_array = None

            poses.append(FramePose(frame_idx=frame_idx, timestamp_ms=timestamp_ms, landmarks=lm_array))
            frame_idx += 1

    cap.release()
    return poses, video_info


def get_wrist_positions(poses: list[FramePose], hand: str = "right") -> np.ndarray:
    """
    Extract (x, y) wrist positions for each frame. NaN where landmarks are missing.
    hand: "right" or "left"
    """
    key = f"{hand}_wrist"
    idx = LM[key]
    positions = np.full((len(poses), 2), np.nan)
    for i, fp in enumerate(poses):
        if fp.landmarks is not None and fp.landmarks[idx, 3] > 0.5:
            positions[i] = fp.landmarks[idx, :2]
    return positions


def get_landmark_positions(poses: list[FramePose], landmark_name: str) -> np.ndarray:
    """Extract (x, y) for a named landmark across all frames."""
    idx = LM[landmark_name]
    positions = np.full((len(poses), 2), np.nan)
    for i, fp in enumerate(poses):
        if fp.landmarks is not None and fp.landmarks[idx, 3] > 0.5:
            positions[i] = fp.landmarks[idx, :2]
    return positions
