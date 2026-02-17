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
