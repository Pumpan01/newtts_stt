# นำเข้า Library ที่ใช้สำหรับ Azure Speech, เสียง, ตัวแปรแวดล้อม และอื่น ๆ
import azure.cognitiveservices.speech as speechsdk
import sounddevice as sd
import numpy as np
import threading
import time
import os
from dotenv import load_dotenv
from pathlib import Path


# ---------- Load API KEY ----------
# โหลด API Key และ Region จากไฟล์ .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")


# ---------- Config ----------
SAMPLE_RATE = 16000        # ความถี่ตัวอย่างเสียง 16kHz
FRAME_DURATION = 30        # ความยาวแต่ละ frame 30ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)  # คำนวณขนาด frame


# ---------- Signal ----------
stop_signal = threading.Event()       # Signal สั่งหยุด STT
cancel_signal = threading.Event()     # Signal แจ้งว่ามีเสียงพูดแทรก
start_vad_signal = threading.Event()  # Signal เริ่มตรวจจับเสียง (VAD)
intent_result = None                  # เก็บ intent ผลลัพธ์ (interrupt หรือ question)


# ---------- Intent Check ----------
# ฟังก์ชันตรวจสอบว่าประโยคเป็นคำถามหรือไม่
def check_intent(text):
    question_keywords = ["อะไร", "ทำไม", "เมื่อไหร่", "ยังไง", "ที่ไหน", "ใคร", "ใช่ไหม", "หรือเปล่า", "กี่", "ไหม"]
    for word in question_keywords:
        if word in text:
            return "question"   # ถ้ามีคำที่บ่งบอกว่าเป็นคำถาม
    return "interrupt"          # ถ้าไม่มี → ถือว่าเป็นการพูดแทรก


# ---------- STT Detect (Continuous) ----------
# ฟังก์ชันเปิดไมค์ฟังเสียงแบบต่อเนื่อง
def detect_human_voice_by_stt_continuous():
    global intent_result

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.speech_recognition_language = "th-TH"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # เมื่อฟังได้ข้อความ
    def recognized(evt):
        global intent_result
        if start_vad_signal.is_set():   # ตรวจเฉพาะตอนเปิด VAD
            text = evt.result.text.strip()
            if text:
                print(f"🛑 ตรวจพบเสียงพูด: {text}")
                intent_result = check_intent(text)  # เช็ค intent
                cancel_signal.set()                 # แจ้งว่ามีเสียงพูดแทรก

    def canceled(evt):
        print(f"❌ STT ยกเลิก: {evt}")

    speech_recognizer.recognized.connect(recognized)
    speech_recognizer.canceled.connect(canceled)

    speech_recognizer.start_continuous_recognition()

    while not stop_signal.is_set():    # ฟังเรื่อย ๆ จนกว่าจะมี stop_signal
        time.sleep(0.1)

    speech_recognizer.stop_continuous_recognition()


# ---------- TTS ----------
# ฟังก์ชันพูดข้อความพร้อมตรวจจับว่ามีเสียงพูดแทรกหรือไม่
def tts_with_cancel_on_speech(text):
    global intent_result

    if not AZURE_SPEECH_KEY or not AZURE_REGION:
        print("❌ กรุณาใส่คีย์ใน .env")
        return

    # เคลียร์ signal ก่อนเริ่ม
    cancel_signal.clear()
    start_vad_signal.clear()
    intent_result = None

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
    )

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()  # แปลงข้อความเป็นเสียง

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        speech_config.speech_synthesis_voice_name = "th-TH-NiwatNeural"
        print("❌ ผิดพลาด:", speechsdk.CancellationDetails.from_result(result).error_details)
        return

    audio_data = result.audio_data
    audio_np = np.frombuffer(audio_data, dtype=np.int16)  # แปลงเป็น numpy array

    audio_duration = len(audio_np) / SAMPLE_RATE
    print(f"🕒 ความยาวเสียง {audio_duration:.2f} วินาที")

    chunk_size = int(SAMPLE_RATE * 0.3)  # แบ่งเล่นทีละ 0.3 วินาที

    print("▶️ เริ่มพูด...")

    # เริ่ม Thread ฟังเสียง
    stop_signal.clear()
    cancel_signal.clear()
    start_vad_signal.set()
    intent_result = None

    mic_thread = threading.Thread(target=detect_human_voice_by_stt_continuous)
    mic_thread.start()

    start_time = time.time()

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        i = 0
        while i < len(audio_np):
            if cancel_signal.is_set():  # ถ้ามีเสียงพูดแทรก → หยุดพูด
                print("🛑 หยุดพูดเพราะมีเสียงพูดแทรก")
                break

            end = min(i + chunk_size, len(audio_np))
            stream.write(audio_np[i:end])  # เล่นเสียง
            i = end

    elapsed_time = time.time() - start_time
    print(f"🕒 ใช้เวลาพูดจริง {elapsed_time:.2f} วินาที")

    start_vad_signal.clear()  # ปิด VAD
    stop_signal.set()         # สั่งหยุดฟัง
    mic_thread.join()         # รอ Thread ฟังเสียงจบ

    # ถ้ามีเสียงแทรก
    if cancel_signal.is_set():
        if intent_result == "interrupt":
            print("👉 เป็นแค่การแทรก → พูดต่อจากที่หยุด")
            remaining_audio = audio_np[i:]
            if len(remaining_audio) > 0:
                tts_with_cancel_on_speech_from_audio(remaining_audio)  # พูดต่อ
            else:
                print("✅ ไม่มีเนื้อหาค้าง → จบแล้ว")

        elif intent_result == "question":
            print("👉 เป็นคำถาม → หยุดเล่าแล้วไปตอบคำถาม")
            print("🤖 [AI] → ตอบคำถามตรงนี้")

    else:
        print("✅ พูดจนจบ")


# 🔥 ฟังก์ชันพูดต่อจากเสียงที่เหลือ
def tts_with_cancel_on_speech_from_audio(audio_np):
    chunk_size = int(SAMPLE_RATE * 0.3)

    stop_signal.clear()
    cancel_signal.clear()
    start_vad_signal.set()
    global intent_result
    intent_result = None

    mic_thread = threading.Thread(target=detect_human_voice_by_stt_continuous)
    mic_thread.start()

    start_time = time.time()

    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        i = 0
        while i < len(audio_np):
            if cancel_signal.is_set():
                print("🛑 หยุดพูดเพราะมีเสียงพูดแทรก")
                break

            end = min(i + chunk_size, len(audio_np))
            stream.write(audio_np[i:end])
            i = end

    elapsed_time = time.time() - start_time
    print(f"🕒 ใช้เวลาพูดต่อ {elapsed_time:.2f} วินาที")

    start_vad_signal.clear()
    stop_signal.set()
    mic_thread.join()

    if cancel_signal.is_set():
        if intent_result == "interrupt":
            print("👉 เป็นแค่การแทรก → พูดต่อจากที่หยุด")
            remaining_audio = audio_np[i:]
            if len(remaining_audio) > 0:
                tts_with_cancel_on_speech_from_audio(remaining_audio)
            else:
                print("✅ ไม่มีเนื้อหาค้าง → จบแล้ว")

        elif intent_result == "question":
            print("👉 เป็นคำถาม → หยุดเล่าแล้วไปตอบคำถาม")
            print("🤖 [AI] → ตอบคำถามตรงนี้")
    else:
        print("✅ พูดจนจบ")


# ---------- Main ----------
if __name__ == "__main__":
    # ข้อความตัวอย่างสำหรับทดสอบ
    text = "นี่คือระบบพูดด้วยเสียงจาก Azure หากมีเสียงพูดแทรก ระบบจะหยุดพูดทันที ถ้าเป็นคำถาม ระบบจะตอบคำถาม แต่ถ้าแค่พูดแทรก ระบบจะพูดต่อจากเดิม"

    stop_signal.clear()
    cancel_signal.clear()

    try:
        tts_with_cancel_on_speech(text)  # เรียกฟังก์ชันพูดหลัก
    finally:
        stop_signal.set()
        print("🛑 จบการทำงาน")
