import pyaudio
import json
from vosk import Model, KaldiRecognizer

# 1. Download a model from https://alphacephei.com/vosk/models
# 2. Extract it and rename the folder to "model"
model_stt = Model("model")
rec = KaldiRecognizer(model_stt, 16000)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Listening...")

# ... (Imports and setup remain the same) ...

print("Listening... (Say 'stop recording' to end)")
text=""
while True:
    data = stream.read(4000)

    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        text +=str(" ") + result['text']
        print(f"Text: {text}")

        # --- ADD THIS CODE HERE ---
        if "stop recording" in text:
            print("Stopping...")
            break
        # --------------------------

# Good practice: Clean up resources after the loop ends
stream.stop_stream()
stream.close()
p.terminate()


import ollama

ollama_client = ollama.Client()
model = "llama3.2"
prompt=text[:-14] + " give the answer in summarized version"
ollama_response = ollama_client.generate(model=model, prompt=prompt)
print("response from ollama:")
print(ollama_response.response)

import pyttsx3

# Initialize TTS engine
engine = pyttsx3.init()

# Convert generated text to audio using the TTS model
output_audio = engine.say(ollama_response.response)
engine.runAndWait()
