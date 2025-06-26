import azure.cognitiveservices.speech as speechsdk
import requests
import os
import shutil
import subprocess
import html
import re

AZURE_SPEECH_KEY = "F7LohbW2EaI1JKreS1P9QxlcpM8K2Y09PPLq9eMp0cUITCPzvCuEJQQJ99BEACqBBLyXJ3w3AAAYACOGqVT5"
AZURE_REGION = "southeastasia"
VOICE_NAME = "th-TH-AcharaNeural"  # th-TH-PremwadeeNeural ,th-TH-NiwatNeural , en-US-RyanMultilingualNeural

LOCAL_WAV_PATH = "D:/TTSPYthon/audio/autoplay.wav"
CONVERTED_WAV_PATH = "D:/TTSPYthon/audio/autoplay_converted.wav"
A2F_AUDIO_DIR = "D:/Omniverse"
A2F_PLAYER_PATH = "/World/audio2face/Player"

def prepare_ssml_text(text):
    safe_text = html.escape(text.strip())
    safe_text = re.sub(r'([.!?])', r'\1<break time="400ms"/>', safe_text)
    safe_text = re.sub(r'\s{2,}', r'<break time="300ms"/>', safe_text)
    return safe_text

def tts_with_emotion(text, emotion="neutral"):
    print(f"สร้างเสียงพร้อมอารมณ์: {emotion}")
    emotion_presets = {
        "neutral":  {"rate": "default", "pitch": "+0st"},
        "happy":    {"rate": "110%", "pitch": "+2st"},
        "sad":      {"rate": "90%",  "pitch": "-1st"},
        "angry":    {"rate": "100%", "pitch": "+1st"},
        "serious":  {"rate": "95%",  "pitch": "-1st"},
        "excited":  {"rate": "110%", "pitch": "+3st"},
        "fear":     {"rate": "98%",  "pitch": "-1st"},
    }

    emo = emotion_presets.get(emotion, emotion_presets["neutral"])
    rate = emo["rate"]
    pitch = emo["pitch"]

    ssml_text = prepare_ssml_text(text)

    ssml = f"""<speak version=\"1.0\" xml:lang=\"th-TH\"><voice name=\"{VOICE_NAME}\"><prosody rate=\"{rate}\" pitch=\"{pitch}\">{ssml_text}</prosody></voice></speak>"""
    print("\U0001F4C4 SSML ที่ส่ง:", ssml)

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=LOCAL_WAV_PATH)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        if result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            print("❌ ไม่สามารถสร้างเสียง:", cancellation.reason)
            if cancellation.reason == speechsdk.CancellationReason.Error:
                print("🛑 Error details:", cancellation.error_details)
        return

    print("✅ สร้างเสียงสำเร็จ:", LOCAL_WAV_PATH)
    convert_to_a2f_format(LOCAL_WAV_PATH, CONVERTED_WAV_PATH)

    # ✅ เปิดโหมด Streaming และ Auto-Generate Emotion
    enable_emotion_streaming()
    enable_auto_generate_emotion()

    copy_and_send_to_a2f(CONVERTED_WAV_PATH)

def convert_to_a2f_format(input_path, output_path):
    print("\U0001F501 แปลงไฟล์ด้วย ffmpeg...")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "44100", "-sample_fmt", "s16", output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ แปลงไฟล์เรียบร้อย:", output_path)

def copy_and_send_to_a2f(filepath):
    filename = os.path.basename(filepath)
    dest_path = os.path.join(A2F_AUDIO_DIR, filename)
    os.makedirs(A2F_AUDIO_DIR, exist_ok=True)
    shutil.copy2(filepath, dest_path)
    print("📁 คัดลอกไฟล์ไปยัง A2F:", dest_path)

    url_set = "http://localhost:8011/A2F/Player/SetTrack"
    payload_set = { "a2f_player": A2F_PLAYER_PATH, "file_name": filename, "time_range": [0, -1] }
    response_set = requests.post(url_set, json=payload_set)
    if response_set.ok:
        print("✅ โหลดเข้า Player:", filename)

        url_loop = "http://localhost:8011/A2F/Player/SetLooping"
        payload_loop = { "a2f_player": A2F_PLAYER_PATH, "loop_audio": False }
        requests.post(url_loop, json=payload_loop)

        url_play = "http://localhost:8011/A2F/Player/Play"
        payload_play = { "a2f_player": A2F_PLAYER_PATH }
        response_play = requests.post(url_play, json=payload_play)

        if response_play.ok:
            print("▶️ A2F เริ่มเล่นเสียง (ไม่วน)")
        else:
            print("❌ เล่นเสียงไม่สำเร็จ:", response_play.status_code, response_play.text)
    else:
        print("❌ โหลดเสียงไม่สำเร็จ:", response_set.status_code, response_set.text)

# ✅ ฟังก์ชันเปิดโหมด Streaming Emotion
def enable_emotion_streaming():
    url = "http://localhost:8011/A2F/A2E/EnableStreaming"
    payload = {
        "a2f_instance": A2F_PLAYER_PATH,
        "enable": True  # ✅ ต้องใส่ค่า enable ด้วย
    }
    response = requests.post(url, json=payload)
    if response.ok:
        print("✅ เปิดโหมด Emotion Streaming")
    else:
        print("❌ เปิด Emotion Streaming ไม่สำเร็จ:", response.status_code, response.text)


# ✅ ฟังก์ชันเปิด Auto-Generate Emotion จากเสียง
def enable_auto_generate_emotion():
    url = "http://localhost:8011/A2F/A2E/EnableAutoGenerateOnTrackChange"
    payload = { "a2f_instance": A2F_PLAYER_PATH, "enable": False }
    response = requests.post(url, json=payload)
    if response.ok:
        print("✅ เปิด Auto-Generate Emotion จากเสียง")
    else:
        print("❌ เปิด Auto-Generate Emotion ไม่สำเร็จ:", response.status_code, response.text)

 