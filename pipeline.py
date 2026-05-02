"""
Golf Swing Analysis Pipeline
=============================

Usage:
    python pipeline.py --video practice.mp4 [options]

Options:
    --video            Path to the input video file (required)
    --hand             Lead wrist to track: "right" (default) or "left"
    --output_dir       Directory for output clips and plots (default: ./output)
    --skip_frames      Process every N+1 frames for speed (default: 0 = every frame)
    --save_clips       Export each swing as a separate MP4 clip
    --no_plot          Skip generating analysis plots
    --no_compose       Skip composing the final annotated video
    --card_duration    Seconds to show each comment card (default: 3.5)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pose_extractor import extract_poses, get_wrist_positions
from swing_segmenter import detect_swings, SwingSegment
from swing_analyzer import analyze_swing, print_analysis, SwingAnalysis
from video_composer import compose_video


# ── Video clip export ────────────────────────────────────────────────────────

def save_swing_clip(video_path: str, segment: SwingSegment, output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, segment.start_frame)
    for fi in range(segment.start_frame, segment.end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        # Annotate frame number
        label = f"Swing #{segment.swing_id + 1}  frame {fi}"
        if fi == segment.impact_frame:
            label += "  [IMPACT]"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()


# ── Plot generation ──────────────────────────────────────────────────────────

def plot_swing_summary(
    analysis: SwingAnalysis,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"スイング #{analysis.swing_id + 1}  (フレーム {analysis.start_frame}–{analysis.end_frame})",
                 fontsize=13)

    # ─ Timing bar chart ─
    ax = axes[0]
    phases = ["バックスイング", "ダウンスイング", "フォロースルー"]
    durations = [
        analysis.timings.address_to_top_sec,
        analysis.timings.top_to_impact_sec,
        analysis.timings.impact_to_finish_sec,
    ]
    colors = ["#4C9BE8", "#E87B4C", "#4CE878"]
    bars = ax.bar(phases, durations, color=colors)
    ax.bar_label(bars, fmt="%.2fs", padding=3)
    ax.set_ylabel("秒")
    ax.set_title("フェーズ時間")
    ax.set_ylim(0, max(durations) * 1.3 + 0.1)

    # ─ Rotation radar (simplified as bar chart) ─
    ax = axes[1]
    metrics = ["肩の回転", "腰の回転", "X-factor"]
    values = [
        analysis.rotation.shoulder_rotation_deg,
        analysis.rotation.hip_rotation_deg,
        analysis.rotation.x_factor_deg,
    ]
    ideal = [90, 45, 45]
    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w / 2, values, w, label="実測値", color="#4C9BE8")
    ax.bar(x + w / 2, ideal, w, label="目標値", color="#E8D44C", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylabel("度 (°)")
    ax.set_title("回転角度")
    ax.legend(fontsize=8)

    # ─ Stability ─
    ax = axes[2]
    stab_labels = ["頭の移動\n(×100)", "脊柱角\nばらつき (°)"]
    stab_vals = [
        analysis.stability.head_displacement_px * 100,
        analysis.stability.spine_angle_std_deg,
    ]
    stab_ideal = [3.0, 5.0]
    bars2 = ax.bar(stab_labels, stab_vals, color=["#9B4CE8", "#E84C9B"])
    ax.bar_label(bars2, fmt="%.2f", padding=3)
    for xi, iv in enumerate(stab_ideal):
        ax.axhline(iv, color="gray", linestyle="--", linewidth=1,
                   label="目標" if xi == 0 else "_")
    ax.set_ylabel("値")
    ax.set_title("安定性")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_speed_timeline(
    all_poses,
    segments: list[SwingSegment],
    fps: float,
    output_path: str,
    hand: str = "right",
) -> None:
    from scipy.ndimage import gaussian_filter1d

    positions = get_wrist_positions(all_poses, hand=hand)
    speed = np.zeros(len(positions))
    for i in range(1, len(positions)):
        if not (np.any(np.isnan(positions[i])) or np.any(np.isnan(positions[i - 1]))):
            speed[i] = np.linalg.norm(positions[i] - positions[i - 1]) * fps
    smooth = gaussian_filter1d(speed, sigma=3.0)

    frames = [p.frame_idx for p in all_poses]
    times = [fi / fps for fi in frames]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(times, smooth, linewidth=1.2, color="#4C9BE8", label="手首速度（平滑化）")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(segments), 1)))
    for seg, col in zip(segments, colors):
        t_start = seg.start_frame / fps
        t_end = seg.end_frame / fps
        t_impact = seg.impact_frame / fps
        ax.axvspan(t_start, t_end, alpha=0.15, color=col, label=f"スイング #{seg.swing_id + 1}")
        ax.axvline(t_impact, color=col, linestyle="--", linewidth=1)

    ax.set_xlabel("時間 (s)")
    ax.set_ylabel("速度 (正規化)")
    ax.set_title("手首速度とスイング区間")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)


# ── JSON report ──────────────────────────────────────────────────────────────

def build_json_report(analyses: list[SwingAnalysis], video_info: dict) -> dict:
    swings = []
    for a in analyses:
        swings.append({
            "swing_id": a.swing_id + 1,
            "frames": {"start": a.start_frame, "end": a.end_frame, "impact": a.impact_frame, "top": a.top_frame},
            "peak_wrist_speed": round(a.peak_wrist_speed, 4),
            "timings": {
                "backswing_sec": round(a.timings.address_to_top_sec, 3),
                "downswing_sec": round(a.timings.top_to_impact_sec, 3),
                "followthrough_sec": round(a.timings.impact_to_finish_sec, 3),
                "tempo_ratio": round(a.timings.backswing_downswing_ratio, 2),
            },
            "rotation_deg": {
                "shoulder": round(a.rotation.shoulder_rotation_deg, 1),
                "hip": round(a.rotation.hip_rotation_deg, 1),
                "x_factor": round(a.rotation.x_factor_deg, 1),
            },
            "stability": {
                "head_displacement": round(a.stability.head_displacement_px, 4),
                "spine_angle_std_deg": round(a.stability.spine_angle_std_deg, 1),
            },
            "feedback": a.feedback,
        })
    return {"video_info": video_info, "swings": swings}


# ── Main pipeline ────────────────────────────────────────────────────────────

def run(
    video_path: str,
    hand: str = "right",
    output_dir: str = "output",
    skip_frames: int = 0,
    save_clips: bool = True,
    make_plots: bool = True,
    make_composed: bool = True,
    card_duration_sec: float = 3.5,
) -> list[SwingAnalysis]:
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/5] ポーズ抽出中: {video_path}")
    poses, video_info = extract_poses(video_path, skip_frames=skip_frames)
    fps = video_info["fps"]
    print(f"      {len(poses)} フレーム抽出完了  ({video_info['total_frames']} フレーム中)")

    print("[2/5] スイング区間の検出中...")
    segments = detect_swings(poses, fps=fps, hand=hand)
    print(f"      {len(segments)} スイングを検出しました")

    if not segments:
        print("      スイングが検出されませんでした。動画を確認してください。")
        return []

    # Speed timeline plot
    if make_plots:
        timeline_path = os.path.join(output_dir, "speed_timeline.png")
        plot_speed_timeline(poses, segments, fps, timeline_path, hand=hand)
        print(f"      タイムラインプロット保存: {timeline_path}")

    print("[3/5] 各スイングを解析中...")
    analyses: list[SwingAnalysis] = []
    for seg in segments:
        try:
            analysis = analyze_swing(seg, poses, fps=fps, hand=hand)
            analyses.append(analysis)
            print_analysis(analysis)

            if make_plots:
                plot_path = os.path.join(output_dir, f"swing_{seg.swing_id + 1:02d}_analysis.png")
                plot_swing_summary(analysis, plot_path)

        except Exception as e:
            print(f"      スイング #{seg.swing_id + 1} の解析でエラー: {e}", file=sys.stderr)

    print("\n[4/5] 結果を保存中...")

    # JSON report
    report = build_json_report(analyses, video_info)
    report_path = os.path.join(output_dir, "swing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"      JSON レポート: {report_path}")

    # Individual clip export
    if save_clips:
        for seg in segments:
            clip_path = os.path.join(output_dir, f"swing_{seg.swing_id + 1:02d}.mp4")
            save_swing_clip(video_path, seg, clip_path)
            print(f"      クリップ保存: {clip_path}")

    # Composed video: comment cards + annotated swings concatenated
    print("\n[5/5] 合成動画を作成中...")
    if make_composed and analyses:
        composed_path = os.path.join(output_dir, "composed_analysis.mp4")
        compose_video(
            video_path=video_path,
            analyses=analyses,
            output_path=composed_path,
            fps=fps,
            card_duration_sec=card_duration_sec,
        )
    else:
        print("      合成動画の作成をスキップしました。")

    print("\n解析完了。")
    return analyses


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Golf Swing Analysis Pipeline")
    parser.add_argument("--video", required=True, help="入力動画ファイルのパス")
    parser.add_argument("--hand", default="right", choices=["right", "left"],
                        help="リード手首: right (デフォルト) または left")
    parser.add_argument("--output_dir", default="output", help="出力ディレクトリ")
    parser.add_argument("--skip_frames", type=int, default=0,
                        help="処理をN+1フレームごとにする（0=全フレーム）")
    parser.add_argument("--save_clips", action="store_true", default=True,
                        help="スイングごとにMP4クリップを保存")
    parser.add_argument("--no_clips", action="store_true",
                        help="クリップ保存をスキップ")
    parser.add_argument("--no_plot", action="store_true",
                        help="グラフ生成をスキップ")
    parser.add_argument("--no_compose", action="store_true",
                        help="合成動画の作成をスキップ")
    parser.add_argument("--card_duration", type=float, default=3.5,
                        help="コメントカードの表示秒数 (デフォルト: 3.5)")
    args = parser.parse_args()

    run(
        video_path=args.video,
        hand=args.hand,
        output_dir=args.output_dir,
        skip_frames=args.skip_frames,
        save_clips=not args.no_clips,
        make_plots=not args.no_plot,
        make_composed=not args.no_compose,
        card_duration_sec=args.card_duration,
    )


if __name__ == "__main__":
    main()
