import streamlit as st
import queue
import time
import io
import wave
import pygame
import pyaudio
import os
import audioop

# --- NEW: Groq Import ---
from groq import Groq

# --- Google STT Import (Still used for Microphone input) ---
from google.cloud import speech
from google.oauth2 import service_account

# --- LangChain Imports ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
RATE = 16000
CHUNK = int(RATE / 10)
GROQ_SAMPLE_RATE = 24000  # Standard for PlayAI/Groq TTS
SILENCE_LIMIT = 1.5  # Fast silence detection

st.set_page_config(page_title="Groq Voice AI", page_icon="⚡")

# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    json_path = st.text_input("Google STT JSON Path", value="speech_totext.json")

    # CHANGED: Groq API Key Input
    groq_api_key = st.text_input("Groq API Key", type="password",value="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")

    ollama_model = st.selectbox("Ollama Model", ["qwen2.5:1.5b", "llama3.2"], index=0)

    # CHANGED: Groq / PlayAI Voices
    voice_choice = st.selectbox("Groq Voice", [
        "Fritz-PlayAI",  # Male, energetic
        "Celeste-PlayAI",  # Female, soft
        "Deedee-PlayAI",  # Female, clear
        "Briggs-PlayAI",  # Male, deep
        "Klara-PlayAI"  # Female, narrator
    ])

    st.markdown("---")
    SILENCE_THRESHOLD = st.slider("Silence Threshold", 100, 3000, 500)


# --- Helper: Convert Raw PCM to WAV ---
def pcm_to_wav_bytes(pcm_data):
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(GROQ_SAMPLE_RATE)
        wav_file.writeframes(pcm_data)
    wav_io.seek(0)
    return wav_io.read()


# --- Audio Playback ---
def play_audio_locally(audio_content):
    try:
        # Note: Groq returns a complete WAV/MP3 file usually, not raw PCM.
        # So we can play it directly without pcm_to_wav_bytes conversion if format is correct.

        pygame.mixer.init()
        audio_stream = io.BytesIO(audio_content)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        st.error(f"Audio Error: {e}")


# --- STT Client (Google) ---
@st.cache_resource
def init_stt_client(api_key_path):
    try:
        creds = service_account.Credentials.from_service_account_file(api_key_path)
        return speech.SpeechClient(credentials=creds)
    except Exception as e:
        st.error(f"Error loading STT JSON: {e}")
        return None


# --- NEW: TTS Client (Groq) ---
def generate_groq_audio(text, api_key, voice_name="Fritz-PlayAI"):
    if not api_key:
        st.error("Missing Groq API Key")
        return None

    try:
        client = Groq(api_key=api_key)

        # Groq TTS Endpoint (OpenAI Compatible)
        response = client.audio.speech.create(
            model="playai-tts",  # The specific PlayAI model
            voice=voice_name,
            input=text,
            response_format="mp3"  # Groq returns MP3 or WAV
        )

        # Return binary content
        return response.content

    except Exception as e:
        st.error(f"Groq TTS Error: {e}")
        return None


# --- Microphone Stream ---
class MicrophoneStream:
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16, channels=1, rate=self._rate, input=True,
            frames_per_buffer=self._chunk, stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        start_silence_time = None

        while not self.closed:
            chunk = self._buff.get()
            if chunk is None: return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None: return
                    data.append(chunk)
                except queue.Empty:
                    break

            audio_bytes = b"".join(data)
            rms = audioop.rms(audio_bytes, 2)

            if rms < SILENCE_THRESHOLD:
                if start_silence_time is None:
                    start_silence_time = time.time()
                if time.time() - start_silence_time > SILENCE_LIMIT:
                    self.closed = True
                    break
            else:
                start_silence_time = None

            yield audio_bytes


# --- Main App ---
st.title("⚡ Groq Voice AI")

stt_client = init_stt_client(json_path)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.button("Start Conversation") and stt_client:
    transcript_placeholder = st.empty()
    status = st.status("Listening...", expanded=True)

    full_transcript = ""
    transcript_parts = []

    try:
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="en-US",
        )
        streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
            responses = stt_client.streaming_recognize(config=streaming_config, requests=requests)

            for response in responses:
                if not response.results: continue
                result = response.results[0]
                if not result.alternatives: continue

                curr = result.alternatives[0].transcript.strip()
                transcript_placeholder.info(f"🗣️ {curr}")

                if result.is_final:
                    transcript_parts.append(curr)

        full_transcript = " ".join(transcript_parts).strip()

    except Exception as e:
        if "generator raised StopIteration" not in str(e):
            st.error(f"Mic Error: {e}")

    if full_transcript:
        status.update(label="Thinking...", state="running")
        transcript_placeholder.empty()

        st.session_state.messages.append({"role": "user", "content": full_transcript})
        st.chat_message("user").write(full_transcript)

        # Chat Generation (Ollama)
        llm = ChatOllama(model=ollama_model, temperature=0.7)
        chain = (
                ChatPromptTemplate.from_messages([
                    ("system", "You are a concise voice assistant. Answer in 1 short sentence."),
                    ("user", "{input}")
                ]) | llm | StrOutputParser()
        )

        ai_text = chain.invoke({"input": full_transcript})

        st.session_state.messages.append({"role": "assistant", "content": ai_text})
        st.chat_message("assistant").write(ai_text)

        # Speak (Groq)
        status.update(label=f"Generating Groq Audio...", state="running")
        audio_data = generate_groq_audio(ai_text, groq_api_key, voice_choice)

        if audio_data:
            status.update(label="Speaking", state="complete")
            play_audio_locally(audio_data)