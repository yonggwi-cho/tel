"""
web_server.py
=============
スマートフォンのブラウザから動画をアップロードして
ゴルフスイング解析を実行するシンプルなWebサーバー。

起動方法:
    python web_server.py
    python web_server.py --host 0.0.0.0 --port 5000

スマートフォンからは http://<このPCのIPアドレス>:5000 にアクセス。
"""

import argparse
import os
import threading
import time
import uuid
from pathlib import Path

from flask import (
    Flask, request, render_template_string, redirect,
    url_for, send_from_directory, jsonify
)

from pipeline import run as run_pipeline

# ── 設定 ────────────────────────────────────────────────────────────────────

UPLOAD_DIR = Path("uploads")
OUTPUT_BASE = Path("output")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v", ".mkv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

# job_id → {"status": "running"|"done"|"error", "message": str, "output_dir": str}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ── HTML テンプレート ────────────────────────────────────────────────────────

BASE_STYLE = """
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body { font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 16px;
         background: #111; color: #eee; }
  h1   { color: #ffd700; font-size: 1.3em; }
  h2   { color: #64b5f6; font-size: 1.1em; margin-top: 24px; }
  a    { color: #90caf9; }
  .card { background: #1e1e1e; border-radius: 10px; padding: 16px; margin: 12px 0; }
  input[type=file] { width: 100%; margin: 8px 0; }
  button, .btn {
    display: inline-block; padding: 12px 24px; border-radius: 8px;
    background: #1565c0; color: #fff; border: none; font-size: 1em;
    cursor: pointer; text-decoration: none; margin-top: 8px; }
  button:active, .btn:active { background: #0d47a1; }
  .ok   { color: #81c784; }
  .err  { color: #ef9a9a; }
  .info { color: #fff176; }
  ul { padding-left: 20px; }
  li { margin: 6px 0; }
  .progress { font-size: 1.2em; text-align: center; padding: 20px; }
</style>
"""

INDEX_HTML = """
<!doctype html><html><head><title>Golf Swing Analyzer</title>""" + BASE_STYLE + """
</head><body>
<h1>⛳ Golf Swing Analyzer</h1>
<div class="card">
  <p>練習動画をアップロードすると、スイングを自動検出して解析します。</p>
  <form method="POST" action="/upload" enctype="multipart/form-data">
    <label>動画ファイル (MP4 / MOV / AVI)</label><br>
    <input type="file" name="video" accept="video/*" required><br>
    <label>利き手（どちらの手が前？）</label><br>
    <select name="hand" style="padding:8px;margin:8px 0;width:100%;background:#333;color:#eee;border-radius:6px;">
      <option value="right">右手前（右利き）</option>
      <option value="left">左手前（左利き）</option>
    </select><br>
    <label>コメントカード表示時間</label><br>
    <select name="card_duration" style="padding:8px;margin:8px 0;width:100%;background:#333;color:#eee;border-radius:6px;">
      <option value="3.0">3秒</option>
      <option value="3.5" selected>3.5秒（デフォルト）</option>
      <option value="5.0">5秒</option>
      <option value="7.0">7秒</option>
    </select><br>
    <button type="submit">📤 アップロードして解析開始</button>
  </form>
</div>
{% if jobs %}
<h2>過去の解析結果</h2>
{% for jid, job in jobs.items() %}
<div class="card">
  <b>{{ job.filename }}</b><br>
  {% if job.status == "done" %}
    <span class="ok">✅ 完了</span>
    <br><a class="btn" href="/results/{{ jid }}">結果を見る</a>
  {% elif job.status == "error" %}
    <span class="err">❌ エラー: {{ job.message }}</span>
  {% else %}
    <span class="info">⏳ 解析中...</span>
    <br><a class="btn" href="/status/{{ jid }}">進捗を確認</a>
  {% endif %}
</div>
{% endfor %}
{% endif %}
</body></html>
"""

STATUS_HTML = """
<!doctype html><html><head><title>解析中...</title>""" + BASE_STYLE + """
<meta http-equiv="refresh" content="4">
</head><body>
<h1>⛳ Golf Swing Analyzer</h1>
<div class="card">
  <h2>解析中: {{ filename }}</h2>
  <div class="progress">⏳ {{ message }}</div>
  <p style="color:#888;font-size:0.85em">このページは4秒ごとに自動更新されます。</p>
</div>
<a href="/">← トップに戻る</a>
</body></html>
"""

RESULTS_HTML = """
<!doctype html><html><head><title>解析結果</title>""" + BASE_STYLE + """
</head><body>
<h1>⛳ 解析結果</h1>
<div class="card">
  <h2>{{ filename }}</h2>
  <p class="ok">✅ {{ n_swings }} スイングを検出・解析しました</p>

  <h2>📹 合成動画（コメント付き）</h2>
  <a class="btn" href="/download/{{ job_id }}/composed_analysis.mp4">
    ⬇ composed_analysis.mp4 をダウンロード
  </a>

  {% if swing_files %}
  <h2>🎞 スイング別クリップ</h2>
  <ul>
    {% for f in swing_files %}
    <li><a href="/download/{{ job_id }}/{{ f }}">{{ f }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}

  {% if plot_files %}
  <h2>📊 解析グラフ</h2>
  <ul>
    {% for f in plot_files %}
    <li><a href="/download/{{ job_id }}/{{ f }}">{{ f }}</a></li>
    {% endfor %}
  </ul>
  {% endif %}

  <h2>📄 JSON レポート</h2>
  <a href="/download/{{ job_id }}/swing_report.json">swing_report.json</a>
</div>
<br><a href="/">← 別の動画を解析する</a>
</body></html>
"""


# ── バックグラウンド解析ジョブ ───────────────────────────────────────────────

def _run_job(job_id: str, video_path: str, hand: str, card_duration: float) -> None:
    output_dir = str(OUTPUT_BASE / job_id)
    with jobs_lock:
        jobs[job_id]["message"] = "ポーズを抽出中..."

    try:
        # ステータスを段階的に更新するため、run() の代わりに個別ステップを呼ぶ
        from pose_extractor import extract_poses
        from swing_segmenter import detect_swings
        from swing_analyzer import analyze_swing
        from video_composer import compose_video
        import json

        with jobs_lock:
            jobs[job_id]["message"] = "ポーズを抽出中... (時間がかかる場合があります)"

        poses, video_info = extract_poses(video_path)
        fps = video_info["fps"]

        with jobs_lock:
            jobs[job_id]["message"] = "スイング区間を検出中..."

        segments = detect_swings(poses, fps=fps, hand=hand)

        with jobs_lock:
            jobs[job_id]["message"] = f"{len(segments)} スイングを検出。解析中..."

        os.makedirs(output_dir, exist_ok=True)
        analyses = []
        for seg in segments:
            analysis = analyze_swing(seg, poses, fps=fps, hand=hand)
            analyses.append(analysis)

        with jobs_lock:
            jobs[job_id]["message"] = "合成動画を生成中..."

        if analyses:
            compose_video(
                video_path=video_path,
                analyses=analyses,
                output_path=os.path.join(output_dir, "composed_analysis.mp4"),
                fps=fps,
                card_duration_sec=card_duration,
            )

            # 個別クリップ
            import cv2
            from swing_segmenter import SwingSegment
            cap = cv2.VideoCapture(video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            for seg in segments:
                clip_path = os.path.join(output_dir, f"swing_{seg.swing_id + 1:02d}.mp4")
                writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start_frame)
                for fi in range(seg.start_frame, seg.end_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                writer.release()
            cap.release()

            # JSON
            report = {
                "video_info": video_info,
                "swings": [
                    {
                        "swing_id": a.swing_id + 1,
                        "frames": {"start": a.start_frame, "end": a.end_frame,
                                   "impact": a.impact_frame, "top": a.top_frame},
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
                    }
                    for a in analyses
                ],
            }
            with open(os.path.join(output_dir, "swing_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        with jobs_lock:
            jobs[job_id].update({
                "status": "done",
                "message": "完了",
                "output_dir": output_dir,
                "n_swings": len(analyses),
            })

    except Exception as e:
        with jobs_lock:
            jobs[job_id].update({"status": "error", "message": str(e)})


# ── ルート ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import render_template_string
    with jobs_lock:
        jobs_copy = dict(jobs)
    return render_template_string(INDEX_HTML, jobs=jobs_copy)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return "ファイルが選択されていません", 400
    if not allowed(f.filename):
        return f"対応していない形式です: {Path(f.filename).suffix}", 400

    hand = request.form.get("hand", "right")
    card_duration = float(request.form.get("card_duration", "3.5"))

    job_id = uuid.uuid4().hex[:12]
    UPLOAD_DIR.mkdir(exist_ok=True)
    save_path = UPLOAD_DIR / f"{job_id}{Path(f.filename).suffix.lower()}"
    f.save(str(save_path))

    with jobs_lock:
        jobs[job_id] = {
            "status": "running",
            "message": "開始中...",
            "filename": f.filename,
            "output_dir": "",
            "n_swings": 0,
        }

    t = threading.Thread(
        target=_run_job,
        args=(job_id, str(save_path), hand, card_duration),
        daemon=True,
    )
    t.start()

    return redirect(url_for("status", job_id=job_id))


@app.route("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return "ジョブが見つかりません", 404
    if job["status"] == "done":
        return redirect(url_for("results", job_id=job_id))
    if job["status"] == "error":
        return render_template_string(
            STATUS_HTML, filename=job["filename"],
            message=f"❌ エラー: {job['message']}"
        )
    return render_template_string(
        STATUS_HTML, filename=job["filename"], message=job["message"]
    )


@app.route("/results/<job_id>")
def results(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return redirect(url_for("status", job_id=job_id))

    out = Path(job["output_dir"])
    swing_files = sorted(p.name for p in out.glob("swing_*.mp4"))
    plot_files  = sorted(p.name for p in out.glob("*.png"))

    return render_template_string(
        RESULTS_HTML,
        job_id=job_id,
        filename=job["filename"],
        n_swings=job["n_swings"],
        swing_files=swing_files,
        plot_files=plot_files,
    )


@app.route("/download/<job_id>/<filename>")
def download(job_id, filename):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return "Not found", 404
    out_dir = job.get("output_dir", "")
    if not out_dir:
        return "Not ready", 404
    # セキュリティ: ファイル名にパス区切りを含めない
    safe_name = Path(filename).name
    return send_from_directory(out_dir, safe_name, as_attachment=True)


# ── 起動 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Golf Swing Analyzer Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="バインドするホスト (デフォルト: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="ポート番号 (デフォルト: 5000)")
    args = parser.parse_args()

    # LAN上のIPアドレスを表示
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "このPCのIPアドレス"

    print("=" * 50)
    print("  Golf Swing Analyzer Web Server 起動中")
    print(f"  スマートフォンのブラウザで以下にアクセス:")
    print(f"  http://{local_ip}:{args.port}")
    print("=" * 50)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
