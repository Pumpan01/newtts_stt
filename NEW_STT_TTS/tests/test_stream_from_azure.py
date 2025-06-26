# นำเข้า Library ที่ใช้
import azure.cognitiveservices.speech as speechsdk
import sounddevice as sd
import numpy as np
import threading
import time
import os
from dotenv import load_dotenv
from pathlib import Path
import random


# ---------- Load API KEY ----------
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION")


# ---------- Config ----------
SAMPLE_RATE = 16000
FRAME_DURATION = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION / 1000)


# ---------- Signal ----------
stop_signal = threading.Event()
cancel_signal = threading.Event()
start_vad_signal = threading.Event()
intent_result = None


# ---------- ฟังก์ชันสุ่ม Emotion และ Gesture ตอนโดนขัด ----------
def get_interrupt_emotion():
    emotions = ["angry", "neutral", "surprised", "sad"]
    return random.choice(emotions)


def get_interrupt_gesture():
    gestures = ["shake_head", "neutral_face", "surprise_face", "sad_face"]
    return random.choice(gestures)


# ---------- Intent Check ----------
def check_intent(text):
    question_keywords = ["อะไร", "ทำไม", "เมื่อไหร่", "ยังไง", "ที่ไหน", "ใคร", "ใช่ไหม", "หรือเปล่า", "กี่", "ไหม"]
    for word in question_keywords:
        if word in text:
            return "question"
    return "interrupt"


# ---------- STT Detect ----------
def detect_human_voice_by_stt_continuous():
    global intent_result

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.speech_recognition_language = "th-TH"
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    def recognized(evt):
        global intent_result
        if start_vad_signal.is_set():
            text = evt.result.text.strip()
            if text:
                print(f"🛑 ตรวจพบเสียงพูด: {text}")
                intent_result = check_intent(text)
                cancel_signal.set()

    def canceled(evt):
        print(f"❌ STT ยกเลิก: {evt}")

    recognizer.recognized.connect(recognized)
    recognizer.canceled.connect(canceled)

    recognizer.start_continuous_recognition()

    while not stop_signal.is_set():
        time.sleep(0.2)

    recognizer.stop_continuous_recognition()


# ---------- TTS ----------
# ---------- TTS ----------

def tts_with_cancel_on_speech(text):
    global intent_result

    if not AZURE_SPEECH_KEY or not AZURE_REGION:
        print("❌ กรุณาใส่คีย์ใน .env")
        return

    cancel_signal.clear()
    start_vad_signal.clear()
    intent_result = None

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
    )

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("❌ ผิดพลาด:", speechsdk.CancellationDetails.from_result(result).error_details)
        return

    audio_data = result.audio_data
    audio_np = np.frombuffer(audio_data, dtype=np.int16)

    audio_duration = len(audio_np) / SAMPLE_RATE
    print(f"🕒 ความยาวเสียง {audio_duration:.2f} วินาที")

    chunk_size = int(SAMPLE_RATE * 0.3)

    print("▶️ เริ่มพูด...")

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
            if cancel_signal.is_set():
                interrupt_emotion = get_interrupt_emotion()
                interrupt_gesture = get_interrupt_gesture()

                print("🛑 หยุดพูดเพราะมีเสียงพูดแทรก")
                print(f"🤖 [Emotion: {interrupt_emotion}] [Gesture: {interrupt_gesture}] → แสดงอารมณ์ตอนโดนขัด")

                if intent_result == "question":
                    print("👉 เป็นคำถาม → หยุดเล่าแล้วไปตอบคำถาม")
                    print("🤖 [AI] → ตอบคำถามตรงนี้")
                    stop_signal.set()
                    start_vad_signal.clear()
                    mic_thread.join()
                    return  # ⛔ หยุดพูดทันทีเพื่อไปตอบ

                elif intent_result == "interrupt":
                    print("👉 เป็นแค่การแทรก → หยุด 2 วินาทีแล้วพูดต่อ")
                    time.sleep(4)  # 🛑 หยุด 2 วินาที
                    cancel_signal.clear()  # 🟩 เคลียร์ signal เพื่อพูดต่อ

            end = min(i + chunk_size, len(audio_np))
            stream.write(audio_np[i:end])
            i = end

    elapsed_time = time.time() - start_time
    print(f"🕒 ใช้เวลาพูดจริง {elapsed_time:.2f} วินาที")

    start_vad_signal.clear()
    stop_signal.set()
    mic_thread.join()

    print("✅ พูดจนจบ")


# ---------- ฟังก์ชันพูดต่อ ----------
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
                interrupt_emotion = get_interrupt_emotion()
                interrupt_gesture = get_interrupt_gesture()

                print("🛑 หยุดพูดเพราะมีเสียงพูดแทรก")
                print(f"🤖 [Emotion: {interrupt_emotion}] [Gesture: {interrupt_gesture}] → แสดงอารมณ์ตอนโดนขัด")
                time.sleep(2)  # ✅ หยุด 2 วินาทีหลังถูกขัด
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
    text = "นี่คือระบบพูดด้วยเสียงจาก Azure หากมีเสียงพูดแทรก ระบบจะหยุดพูดทันที ถ้าเป็นคำถาม ระบบจะตอบคำถาม แต่ถ้าแค่พูดแทรก ระบบจะพูดต่อจากเดิม"

    stop_signal.clear()
    cancel_signal.clear()

    try:
        tts_with_cancel_on_speech(text)
    finally:
        stop_signal.set()
        print("🛑 จบการทำงาน")
