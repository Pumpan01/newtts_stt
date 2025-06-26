#http://127.0.0.1:8000/tts/?text=Use%20the%20NVIDIA%20Audio2Face%20headless%20server%20and%20interact%20with%20it%20through%20a%20requests%20API
#http://192.168.1.105:8000/tts/?text=Use%20the%20NVIDIA%20Audio2Face%20headless%20server%20and%20interact%20with%20it%20through%20a%20requests%20API
#ส่งเสียงผ่าน AZure TTS ไปยัง NVIDIA Audio2Face ผ่าน gRPC Streaming
#เปิด audio2face แบบ gui ไว้เพื่อดูว่าโปรแกรทำอะไรอยู่ แล้วโหลด lib\site-packages\py_audio2face\assets\mark_arkit_solved_streaming.usd

# uvicorn fastapi_tts:app --reload --app-dir tests
# uvicorn fastapi_tts:app --reload --host 192.168.1.105 --port 8000 --app-dir tests
# uvicorn fastapi_tts:app --reload --host 192.168.1.105 --port 8000

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading

# 🔗 นำเข้า function และ signal ต่าง ๆ จาก test_stream_from_azure
from test_stream_from_azure import (
    tts_with_cancel_on_speech,           # ฟังก์ชันเล่นเสียงพูด (TTS) โดยสามารถหยุดกลางทางถ้ามีการแทรก
    detect_human_voice_by_stt_continuous, # ฟังก์ชันเปิดไมค์ฟังเสียง → ใช้ STT ตรวจจับเสียงพูด
    stop_signal,                          # Signal เพื่อสั่งหยุดการฟัง STT
    cancel_signal,                        # Signal ที่บอกว่ามีคนพูดแทรก
    start_vad_signal,                     # Signal ที่สั่งเริ่มระบบ VAD (ตรวจจับเสียงพูด)
    intent_result                          # ตัวแปรเก็บผล Intent เช่น interrupt หรือ question
)

# ✅ สร้าง FastAPI app
app = FastAPI()


# ---------------------------
# 📦 Model สำหรับ POST API
class Item(BaseModel):
    text: str
    emotion: str = "neutral"   # ✅ เพิ่ม emotion
    gesture: str = "neutral_face"  # ✅ เพิ่ม gesture


# ---------------------------
# 🔊 GET Endpoint สำหรับพูด
@app.get("/tts/")
async def tts(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="กรุณาระบุพารามิเตอร์ text")

    intent = run_tts_pipeline(text)    # รัน TTS Pipeline

    return {
        "message": f"TTS for \"{text}\" finished",
        "intent": intent
    }


# ---------------------------
# 🔊 POST Endpoint สำหรับพูด
@app.post("/speak/")
async def speak(item: Item):
    if not item.text:
        raise HTTPException(status_code=400, detail="กรุณาระบุพารามิเตอร์ text")

    print("🔊 ข้อความที่จะพูด:", item.text)
    print("😊 อารมณ์:", item.emotion)
    print("🕺 ท่าทาง:", item.gesture)

    # ✅ พูดเฉพาะ text (ซึ่งคือ answer)
    intent = run_tts_pipeline(item.text)

    return {
        "message": f"TTS for \"{item.text}\" finished",
        "intent": intent
    }


# ---------------------------
# 🏠 เช็คว่า API ทำงานปกติ
@app.get("/")
async def read_root():
    return {"status": "ok", "message": "Azure TTS + STT Interrupt + Intent API is running"}


# ---------------------------
# 🔥 Pipeline หลัก → เล่นเสียงพูด + ตรวจจับว่ามีคนพูดแทรกหรือไม่
def run_tts_pipeline(text):
    global intent_result

    # ✅ เริ่มต้น → เคลียร์ signal ทั้งหมด
    stop_signal.clear()                # ไม่หยุด
    cancel_signal.clear()              # ไม่มีการแทรก
    start_vad_signal.clear()           # ยังไม่เริ่ม VAD
    intent_result = None               # เคลียร์ผล intent

    # ✅ เริ่ม Thread สำหรับ STT ฟังเสียง
    mic_thread = threading.Thread(target=detect_human_voice_by_stt_continuous)
    mic_thread.start()

    try:
        start_vad_signal.set()         # 🔥 สั่งเริ่ม VAD → ตรวจจับเสียงพูดระหว่างเล่น TTS

        print(f"▶️ เริ่มพูด: {text}")
        tts_with_cancel_on_speech(text)  # 🔊 เริ่มพูดข้อความ

        # 🔥 เช็คว่าระหว่างพูด มี cancel_signal เกิดขึ้นไหม (แปลว่ามีคนพูดแทรก)
        if cancel_signal.is_set():
            if intent_result == "interrupt":
                # 👉 เจอการแทรก → หยุดชั่วคราว แล้วพูดข้อความต่อ
                print("👉 เป็นการแทรก → พูดต่อจนจบ")
                remaining_text = text  # ✅ ในอนาคตสามารถปรับให้แบ่งข้อความได้
                tts_with_cancel_on_speech(remaining_text)

            elif intent_result == "question":
                # 👉 ถ้าเป็นคำถาม → หยุดพูด แล้วส่งข้อความต่อไปยัง Chatbot (ยังไม่เขียนในโค้ดนี้)
                print("👉 เป็นคำถาม → หยุดพูดแล้วไปตอบคำถาม")
                print("🤖 [AI] → ตอบคำถามตรงนี้")
                return "question"

        # ✅ จบ → คืนค่าผล intent ถ้ามี, ถ้าไม่มี → คืนค่า 'none'
        return intent_result if intent_result else "none"

    finally:
        # ✅ ไม่ว่าจะเกิดอะไรขึ้น → สั่งหยุด STT และ VAD
        stop_signal.set()
        start_vad_signal.clear()
        mic_thread.join()
        print("🛑 เสร็จสิ้น")



