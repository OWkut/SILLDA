from flask import Flask, render_template, Response, redirect, request, url_for, jsonify
import os
from core.stream_manager import stream_manager

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    videos = sorted(os.listdir(UPLOAD_FOLDER))
    return render_template("index.html", video=None, use_webcam=False, videos=videos)


@app.route("/video/<filename>")
def play_video(filename):
    stream_manager.start_video(os.path.join(UPLOAD_FOLDER, filename))
    videos = sorted(os.listdir(UPLOAD_FOLDER))
    return render_template(
        "index.html", video=filename, use_webcam=False, videos=videos
    )


@app.route("/webcam")
def webcam():
    stream_manager.start_webcam()
    videos = sorted(os.listdir(UPLOAD_FOLDER))
    return render_template("index.html", video=None, use_webcam=True, videos=videos)


@app.route("/video_feed")
def video_feed():
    return Response(
        stream_manager.generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/perf")
def perf():
    try:
        stats = stream_manager.get_fps()
        return jsonify(stats)
    except Exception as e:
        print("Erreur /perf:", e)
        return jsonify({"error": "Erreur interne"}), 500


@app.route("/pause_stream", methods=["POST"])
def pause_stream():
    stream_manager.pause()
    return "Stream paused", 200


@app.route("/resume_stream", methods=["POST"])
def resume_stream():
    stream_manager.resume()
    return "Stream resumed", 200


if __name__ == "__main__":
    app.run(debug=True)
