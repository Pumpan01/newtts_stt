# นำเข้าไลบรารีที่จำเป็น
import cv2                        # OpenCV สำหรับการประมวลผลภาพ
import mediapipe as mp             # MediaPipe สำหรับตรวจจับคนในกล้อง
import azure.cognitiveservices.speech as speechsdk  # Azure Speech SDK สำหรับ STT
import sounddevice as sd           # สำหรับจัดการเสียงจากไมค์
import numpy as np                 # จัดการข้อมูลเสียงแบบ array
import webrtcvad                   # ตรวจจับเสียงพูด/เสียงเงียบ (VAD)
import requests                    # ส่ง request ไป API
import threading                   # รันหลาย thread พร้อมกัน
import time                        # จับเวลา
import os                          # จัดการไฟล์
from dotenv import load_dotenv     # โหลดตัวแปรจาก .env
from pathlib import Path           # จัดการ path
import keyboard                    # ตรวจจับการกดปุ่มบนคีย์บอร์ด


# ---------- โหลด API KEY จากไฟล์ .env ----------
env_path = Path(__file__).resolve().parent / "tests" / ".env"
load_dotenv(dotenv_path=env_path)

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")    # คีย์ของ Azure Speech
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")     # Region ของ Azure Speech
API_ASKDAMO = "http://localhost:3000/0921_chatbot_demo/api/askDamo"  # API สำหรับ chatbot


# ---------- ตั้งค่า Log ----------
LOG_FILE = "log.txt"

def write_log(message):
    print(message)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{time.ctime()}] {message}\n")


# ---------- ตั้งค่า MediaPipe สำหรับตรวจจับคน ----------
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)


# ---------- ตรวจจับความเงียบด้วย webrtcvad ----------
def wait_for_silence(vad_aggressiveness=2, silence_duration=3, sample_rate=16000):
    vad = webrtcvad.Vad(vad_aggressiveness)     # สร้างตัวตรวจจับเสียงพูด
    silence_start = None
    block_duration = 0.03  # 30ms

    def is_speech(audio_chunk):
        audio_bytes = (audio_chunk * 32768).astype(np.int16).tobytes()
        return vad.is_speech(audio_bytes, sample_rate)

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
            while True:
                audio_chunk, _ = stream.read(int(sample_rate * block_duration))
                if is_speech(audio_chunk):
                    silence_start = None  # มีเสียงพูด → รีเซ็ตตัวจับเวลา
                else:
                    if silence_start is None:
                        silence_start = time.time()  # เริ่มจับเวลาเมื่อเจอเสียงเงียบ
                    elif time.time() - silence_start >= silence_duration:
                        break  # ถ้าเงียบต่อเนื่องเกิน silence_duration → หยุด
    except Exception as e:
        write_log(f"❌ Error in VAD: {e}")


# ---------- ฟังก์ชันเปิดไมค์ฟังจนเงียบแล้วส่งไป API ----------
def listen_and_send(stop_event):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.speech_recognition_language = "th-TH"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    while not stop_event.is_set():   # ทำงานจนกว่าจะมีคำสั่งหยุด
        full_text = []

        def recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                write_log(f"📝 ได้ข้อความ: {evt.result.text}")
                full_text.append(evt.result.text)

        recognizer.recognized.connect(recognized)

        write_log("🎙️ เปิดไมค์ (พูดจนเงียบ 2 วิ)")
        recognizer.start_continuous_recognition()

        wait_for_silence(silence_duration=2)  # ฟังจนเงียบ 2 วิ

        recognizer.stop_continuous_recognition()
        recognizer.recognized.disconnect_all()

        text = ' '.join(full_text).strip()

        if text:
            write_log(f"📨 ส่งข้อความไป API: {text}")
            try:
                response = requests.post(API_ASKDAMO, json={"question": text})
                response.raise_for_status()
                write_log("✅ ส่งสำเร็จ รอรอบใหม่...")
            except Exception as e:
                write_log(f"❌ ส่ง API ผิดพลาด: {e}")
        else:
            write_log("⚠️ ไม่มีข้อความ")

        time.sleep(0.1)


# ---------- เริ่มเปิดกล้อง ----------
cap = cv2.VideoCapture(1)  # ใช้กล้องตัวที่ 1 (หรือ 0 แล้วแต่เครื่อง)

mic_on = False                     # สถานะว่าไมค์เปิดไหม
mic_thread = None                  # Thread สำหรับไมค์
stop_event = threading.Event()     # Signal สำหรับหยุดไมค์

write_log("🎉 ยินดีต้อนรับ! พร้อมแล้ว พูดอะไรกับฉันก็ได้เลย 😎")

while cap.isOpened():              # ทำงานจนกว่าจะปิดกล้อง
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = selfie_segmentation.process(frame_rgb)

    condition = results.segmentation_mask > 0.5   # เช็คว่ามีคนในกล้องไหม
    has_person = condition.any()

    if has_person:
        if not mic_on:
            write_log("👀 พบคน → เปิดไมค์")
            stop_event.clear()   # ยกเลิกคำสั่งหยุด
            mic_thread = threading.Thread(target=listen_and_send, args=(stop_event,))
            mic_thread.start()   # เริ่ม thread ฟังเสียง
            mic_on = True
    else:
        if mic_on:
            write_log("❌ ไม่พบคน → รอให้พูดจบก่อน แล้วปิดไมค์")
            stop_event.set()     # สั่งให้หยุดฟัง
            mic_thread.join()    # รอให้ thread จบก่อน
            mic_on = False

    # ถ้ากดปุ่ม ESC → หยุดโปรแกรม
    if keyboard.is_pressed('esc'):
        write_log("👋 ออกจากโปรแกรม")
        if mic_on:
            stop_event.set()
            mic_thread.join()
        break

cap.release()
cv2.destroyAllWindows()
