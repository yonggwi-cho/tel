"""
video_composer.py
=================
Takes individual swing clips (with pose data) and composes a single
annotated video:

  [コメントカード #1] → [スイング #1 フッテージ] → [コメントカード #2] → ...

コメントカードの内容:
  - スイング番号・フレーム範囲
  - タイミング (backswing / downswing / followthrough)
  - 回転 (肩, 腰, X-factor)
  - 安定性 (頭, 脊柱)
  - フィードバック箇条書き

スイング映像フレームには:
  - 右下に小さなスイング番号バッジ
  - インパクトフレームに [IMPACT] 強調バナー
  - バックスイングトップに [TOP] バナー
"""

import cv2
import numpy as np
from dataclasses import dataclass

from swing_analyzer import SwingAnalysis
from swing_segmenter import SwingSegment


# ── 色・フォント設定 ────────────────────────────────────────────────────────

FONT = cv2.FONT_HERSHEY_SIMPLEX
CARD_BG      = (20, 20, 20)
CARD_TITLE   = (255, 215, 0)      # gold
CARD_SECTION = (100, 200, 255)    # light blue
CARD_VALUE   = (220, 220, 220)
CARD_GOOD    = (80, 220, 80)
CARD_WARN    = (80, 180, 255)
CARD_BAD     = (80, 80, 255)


# ── テキスト描画ユーティリティ ──────────────────────────────────────────────

def _put(img, text, x, y, color, scale=0.55, thickness=1):
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _put_bold(img, text, x, y, color, scale=0.7):
    cv2.putText(img, text, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def _semi_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ── コメントカード生成 ─────────────────────────────────────────────────────

def make_comment_card(
    analysis: SwingAnalysis,
    width: int,
    height: int,
    fps: float,
    duration_sec: float = 3.5,
) -> list[np.ndarray]:
    """
    静止した「コメントカード」フレームを duration_sec 秒分生成して返す。
    """
    n_frames = max(1, int(round(duration_sec * fps)))
    frame = np.full((height, width, 3), CARD_BG, dtype=np.uint8)

    # ─ タイトルバー ─
    _semi_rect(frame, 0, 0, width, 60, (40, 40, 80))
    title = f"  Swing #{analysis.swing_id + 1}   " \
            f"frames {analysis.start_frame} - {analysis.end_frame}"
    _put_bold(frame, title, 12, 40, CARD_TITLE, scale=0.75)

    col1_x = 30
    col2_x = width // 2 + 20
    y = 90
    line_h = 28

    def row(label, value, col_x, y, color=CARD_VALUE):
        _put(frame, label, col_x, y, CARD_SECTION, scale=0.52)
        _put(frame, value, col_x + 200, y, color, scale=0.52)

    # ─ タイミング列 ─
    _put_bold(frame, "[ タイミング ]", col1_x, y, CARD_SECTION, scale=0.6)
    y += line_h
    row("バックスイング",  f"{analysis.timings.address_to_top_sec:.2f} s",  col1_x, y)
    y += line_h
    row("ダウンスイング",  f"{analysis.timings.top_to_impact_sec:.2f} s",   col1_x, y)
    y += line_h
    row("フォロースルー",  f"{analysis.timings.impact_to_finish_sec:.2f} s",col1_x, y)
    y += line_h
    ratio = analysis.timings.backswing_downswing_ratio
    ratio_col = CARD_GOOD if 2.5 <= ratio <= 4.0 else CARD_BAD
    row("テンポ比",        f"{ratio:.1f} : 1",                              col1_x, y, ratio_col)

    # ─ 回転列 ─
    y_r = 90
    _put_bold(frame, "[ 回転 ]", col2_x, y_r, CARD_SECTION, scale=0.6)
    y_r += line_h
    sh = analysis.rotation.shoulder_rotation_deg
    row("肩の回転",  f"{sh:.1f}°", col2_x, y_r, CARD_GOOD if sh >= 70 else CARD_BAD)
    y_r += line_h
    hi = analysis.rotation.hip_rotation_deg
    row("腰の回転",  f"{hi:.1f}°", col2_x, y_r, CARD_GOOD if hi >= 35 else CARD_BAD)
    y_r += line_h
    xf = analysis.rotation.x_factor_deg
    row("X-ファクター", f"{xf:.1f}°", col2_x, y_r, CARD_GOOD if xf >= 30 else CARD_WARN)
    y_r += line_h

    # ─ 安定性 ─
    y_r += 6
    _put_bold(frame, "[ 安定性 ]", col2_x, y_r, CARD_SECTION, scale=0.6)
    y_r += line_h
    hd = analysis.stability.head_displacement_px
    row("頭の移動量", f"{hd:.4f}", col2_x, y_r, CARD_GOOD if hd < 0.06 else CARD_BAD)
    y_r += line_h
    sa = analysis.stability.spine_angle_std_deg
    row("脊柱角ばらつき", f"{sa:.1f}°", col2_x, y_r, CARD_GOOD if sa < 8 else CARD_BAD)

    # ─ フィードバック ─
    sep_y = max(y, y_r) + 20
    cv2.line(frame, (20, sep_y), (width - 20, sep_y), (80, 80, 80), 1)
    fb_y = sep_y + 28
    _put_bold(frame, "[ フィードバック ]", col1_x, fb_y, CARD_SECTION, scale=0.6)
    fb_y += line_h
    for tip in analysis.feedback:
        if fb_y + line_h > height - 10:
            break
        _put(frame, f"  * {tip}", col1_x, fb_y, CARD_VALUE, scale=0.48)
        fb_y += line_h

    return [frame.copy() for _ in range(n_frames)]


# ── スイング映像アノテーション ─────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    analysis: SwingAnalysis,
) -> np.ndarray:
    """
    スイング映像の1フレームにオーバーレイを描く。
    - 右下: スイング番号バッジ
    - インパクト: 黄色バナー
    - トップ: 水色バナー
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # 右下バッジ
    badge = f"Swing #{analysis.swing_id + 1}"
    (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.55, 1)
    bx, by = w - bw - 16, h - 12
    _semi_rect(out, bx - 6, by - bh - 4, bx + bw + 6, by + 6, (0, 0, 0), alpha=0.6)
    _put(out, badge, bx, by, (220, 220, 220))

    # バナー
    if frame_idx == analysis.impact_frame:
        _semi_rect(out, 0, h - 50, w, h, (0, 80, 200), alpha=0.7)
        _put_bold(out, "  IMPACT", 10, h - 15, (255, 230, 0), scale=0.8)

    elif frame_idx == analysis.top_frame:
        _semi_rect(out, 0, h - 50, w, h, (0, 130, 100), alpha=0.7)
        _put_bold(out, "  TOP OF BACKSWING", 10, h - 15, (180, 255, 255), scale=0.8)

    return out


# ── メイン合成関数 ─────────────────────────────────────────────────────────

def compose_video(
    video_path: str,
    analyses: list[SwingAnalysis],
    output_path: str,
    fps: float,
    card_duration_sec: float = 3.5,
) -> None:
    """
    コメントカード + アノテーション済みスイング映像を交互に並べて1本の動画にする。

    Parameters
    ----------
    video_path       : 元動画ファイルパス
    analyses         : analyze_swing() の結果リスト
    output_path      : 出力MP4パス
    fps              : フレームレート
    card_duration_sec: コメントカードの表示秒数
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # フレーム全体をメモリに読まず、解析ごとに必要範囲だけシーク
    for analysis in analyses:
        seg_start = analysis.start_frame
        seg_end   = analysis.end_frame

        # 1. コメントカード
        cards = make_comment_card(analysis, width, height, fps, card_duration_sec)
        for card_frame in cards:
            writer.write(card_frame)

        # 2. スイング映像（アノテーション付き）
        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        for fi in range(seg_start, seg_end + 1):
            ret, frame = cap.read()
            if not ret:
                break
            annotated = annotate_frame(frame, fi, analysis)
            writer.write(annotated)

    cap.release()
    writer.release()
    print(f"合成動画を保存しました: {output_path}")
