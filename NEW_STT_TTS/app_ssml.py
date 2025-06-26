# app_ssml.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread
from tts_a2f_ssml import tts_with_emotion 

app = Flask(__name__)
CORS(app)

@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "").strip()
    emotion = data.get("emotion", "neutral").strip().lower()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # ✅ ส่ง emotion เข้าไปด้วย
    Thread(target=tts_with_emotion, args=(text, emotion)).start()

    return jsonify({"status": "OK", "message": f"กำลังพูด: {text} ({emotion})"}), 200


if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000)
