import ollama
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import subprocess
from faster_whisper import WhisperModel

MODEL_NAME = "llama3:latest"
VOICE_MODEL = "voices/en_US-lessac-medium.onnx"
SAMPLE_RATE = 16000
DURATION = 4 

print("Loading Whisper Model")
whisper = WhisperModel("base", compute_type="int8")

SYSTEM_PROMPT = "You are Jarvis, a smart and helpful AI voice assistant."
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

def record_audio():
    print("\n Listening..")
    audio=sd.rec(int(SAMPLE_RATE*DURATION),samplerate=SAMPLE_RATE,channels=1,dtype=np.int16)
    sd.wait()
    wav.write("input.wav",SAMPLE_RATE,audio)
    return "input.wav"
def transcribe(audio_path):
    segments, _ = whisper.transcribe(audio_path)
    text=""
    for j in segments:
        text+=j.text
    return text.strip()

def jarvis(text):
    messages.append({"role":"user","content":text})
    response=ollama.chat(
        model=MODEL_NAME,
        messages=messages
    )
    reply= response["message"]["content"]
    messages.append({"role":"assistant","content":reply})
    return reply
def speak(text):
    subprocess.run([
        "piper","--model",VOICE_MODEL,"--output_file","output.wav"
    ],input=text.encode())
    subprocess.run(["afplay","output.wav"])

print("\n Jarvis voice Assistant Read,press ctrl+c to stop")

try:
    while True:
        audio_file=record_audio()
        user_text=transcribe(audio_file)
        if not user_text:
            continue
        print(f"\nYou said: {user_text}")
        reply=jarvis(user_text)
        print(f"Jarvis: {reply}")
        speak(reply)
except KeyboardInterrupt:
    print("\nJarvis stopped.")

