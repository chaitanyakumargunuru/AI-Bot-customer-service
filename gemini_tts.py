# import streamlit as st
# import queue
# import time
# import io
# import wave
# import pygame
# import pyaudio
# import os
# import audioop  # Required for silence detection
#
# # --- CRITICAL GOOGLE IMPORTS ---
# from google.cloud import speech
# from google.oauth2 import service_account
# from google import genai
# from google.genai import types
#
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
#
# # --- Configuration ---
# RATE = 16000
# CHUNK = int(RATE / 10)
# GEMINI_SAMPLE_RATE = 24000  # Gemini 2.5 Flash default
# SILENCE_LIMIT = 4  # Seconds of silence before stopping
#
# st.set_page_config(page_title="Umbriel Voice AI", page_icon="🌑")
#
# # --- Sidebar ---
# with st.sidebar:
#     st.title("Settings")
#     json_path = st.text_input("Google STT JSON Path", value="speech_totext.json")
#     gemini_api_key = st.text_input("Google AI Studio Key (Gemini)", type="password",value="AIzaSyBMq7Am6PMyRe6GAKNskgcCXxxrVPdPjek")
#     ollama_model = st.selectbox("Ollama Model", ["qwen2.5:1.5b", "llama3.2"], index=0)
#     voice_choice = st.selectbox("Voice", ["Umbriel", "Aoede", "Puck", "Charon", "Fenrir", "Despina"])
#
#     st.markdown("---")
#     SILENCE_THRESHOLD = st.slider("Silence Threshold (Sensitivity)", 100, 3000, 700)
#     st.caption("Lower = more sensitive to background noise. Higher = need to speak louder.")
#
#
# # --- Helper: Convert Raw PCM to WAV ---
# def pcm_to_wav_bytes(pcm_data):
#     """Wraps raw PCM audio bytes into a valid WAV file structure."""
#     wav_io = io.BytesIO()
#     with wave.open(wav_io, "wb") as wav_file:
#         wav_file.setnchannels(1)
#         wav_file.setsampwidth(2)
#         wav_file.setframerate(GEMINI_SAMPLE_RATE)
#         wav_file.writeframes(pcm_data)
#     wav_io.seek(0)
#     return wav_io.read()
#
#
# # --- Audio Playback ---
# def play_audio_locally(audio_bytes):
#     try:
#         # 1. Convert Raw PCM -> WAV
#         wav_data = pcm_to_wav_bytes(audio_bytes)
#
#         # 2. Init Mixer
#         if pygame.mixer.get_init() is None:
#             pygame.mixer.init(frequency=GEMINI_SAMPLE_RATE)
#
#         # 3. Load & Play
#         audio_stream = io.BytesIO(wav_data)
#         pygame.mixer.music.load(audio_stream)
#         pygame.mixer.music.play()
#
#         while pygame.mixer.music.get_busy():
#             pygame.time.Clock().tick(10)
#
#     except Exception as e:
#         st.error(f"Audio Error: {e}")
#
#
# # --- STT Client ---
# @st.cache_resource
# def init_stt_client(api_key_path):
#     try:
#         creds = service_account.Credentials.from_service_account_file(api_key_path)
#         return speech.SpeechClient(credentials=creds)
#     except Exception as e:
#         st.error(f"Error loading STT JSON: {e}")
#         return None
#
#
# # --- TTS Client (Gemini) ---
# def generate_umbriel_audio(text, api_key, voice_name="Umbriel"):
#     if not api_key:
#         st.error("Missing API Key")
#         return None
#
#     try:
#         client = genai.Client(api_key=api_key)
#
#         response = client.models.generate_content(
#             model="gemini-2.5-flash-preview-tts",
#             contents=f"Say this text: {text}",
#             config=types.GenerateContentConfig(
#                 response_modalities=["AUDIO"],
#                 speech_config=types.SpeechConfig(
#                     voice_config=types.VoiceConfig(
#                         prebuilt_voice_config=types.PrebuiltVoiceConfig(
#                             voice_name=voice_name
#                         )
#                     )
#                 )
#             )
#         )
#         return response.parts[0].inline_data.data
#
#     except Exception as e:
#         st.error(f"Umbriel Error: {e}")
#         return None
#
#
# # --- Microphone Stream (With Silence Detection) ---
# class MicrophoneStream:
#     def __init__(self, rate, chunk):
#         self._rate = rate
#         self._chunk = chunk
#         self._buff = queue.Queue()
#         self.closed = True
#
#     def __enter__(self):
#         self._audio_interface = pyaudio.PyAudio()
#         self._audio_stream = self._audio_interface.open(
#             format=pyaudio.paInt16, channels=1, rate=self._rate, input=True,
#             frames_per_buffer=self._chunk, stream_callback=self._fill_buffer,
#         )
#         self.closed = False
#         return self
#
#     def __exit__(self, type, value, traceback):
#         self._audio_stream.stop_stream()
#         self._audio_stream.close()
#         self.closed = True
#         self._buff.put(None)
#         self._audio_interface.terminate()
#
#     def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
#         self._buff.put(in_data)
#         return None, pyaudio.paContinue
#
#     def generator(self):
#         start_silence_time = None
#
#         while not self.closed:
#             # Get data from buffer
#             chunk = self._buff.get()
#             if chunk is None: return
#             data = [chunk]
#
#             while True:
#                 try:
#                     chunk = self._buff.get(block=False)
#                     if chunk is None: return
#                     data.append(chunk)
#                 except queue.Empty:
#                     break
#
#             audio_bytes = b"".join(data)
#
#             # --- Silence Detection Logic ---
#             rms = audioop.rms(audio_bytes, 2)
#
#             # Check if volume is below threshold
#             if rms < SILENCE_THRESHOLD:
#                 if start_silence_time is None:
#                     start_silence_time = time.time()
#
#                 # If silent for more than 4 seconds, break the loop
#                 if time.time() - start_silence_time > SILENCE_LIMIT:
#                     self.closed = True
#                     break
#             else:
#                 # Reset timer if sound is detected
#                 start_silence_time = None
#
#             yield audio_bytes
#
#
# # --- Main App ---
# st.title("🎙️ Voice AI: Umbriel Edition")
#
# stt_client = init_stt_client(json_path)
#
# # Initialize Session State for chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []
#
# # Display Chat History
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# if st.button("Start Conversation") and stt_client:
#     transcript_placeholder = st.empty()
#     status = st.status("Listening...", expanded=True)
#
#     full_transcript = ""
#     transcript_parts = []
#
#     # 1. Listen (With Silence Detection)
#     try:
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=RATE,
#             language_code="en-US",
#         )
#         streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)
#
#         with MicrophoneStream(RATE, CHUNK) as stream:
#             audio_generator = stream.generator()
#             requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
#             responses = stt_client.streaming_recognize(config=streaming_config, requests=requests)
#
#             for response in responses:
#                 if not response.results: continue
#                 result = response.results[0]
#                 if not result.alternatives: continue
#
#                 # Get current transcript snippet
#                 curr = result.alternatives[0].transcript.strip()
#
#                 # Live Display on Screen
#                 transcript_placeholder.info(f"🗣️ {curr}")
#
#                 if result.is_final:
#                     transcript_parts.append(curr)
#
#         # Combine all parts
#         full_transcript = " ".join(transcript_parts).strip()
#
#     except Exception as e:
#         # Ignore stream closed error (happens naturally when we break loop)
#         if "generator raised StopIteration" not in str(e):
#             st.error(f"Mic Error: {e}")
#
#     # 2. Process (Only if something was said)
#     if full_transcript:
#         status.update(label="Thinking...", state="running")
#         transcript_placeholder.empty()  # Clear the "Listening" info box
#
#         # Add User input to history
#         st.session_state.messages.append({"role": "user", "content": full_transcript})
#         st.chat_message("user").write(full_transcript)
#
#         # LangChain / Ollama Processing
#         llm = ChatOllama(model=ollama_model, temperature=0.7)
#         chain = (
#                 ChatPromptTemplate.from_messages([
#                     ("system", "You are a helpful assistant. Be concise."),
#                     ("user", "{input}")
#                 ]) | llm | StrOutputParser()
#         )
#
#         ai_text = chain.invoke({"input": full_transcript})
#
#         # Add AI response to history
#         st.session_state.messages.append({"role": "assistant", "content": ai_text})
#         st.chat_message("assistant").write(ai_text)
#
#         # 3. Speak (Umbriel)
#         status.update(label=f"Generating {voice_choice} Audio...", state="running")
#         audio_data = generate_umbriel_audio(ai_text, gemini_api_key, voice_choice)
#
#         if audio_data:
#             status.update(label="Speaking", state="complete")
#             play_audio_locally(audio_data)


import streamlit as st
import queue
import time
import io
import wave
import pygame
import pyaudio
import os
import audioop

# --- CRITICAL GOOGLE IMPORTS ---
from google.cloud import speech
from google.oauth2 import service_account
from google import genai
from google.genai import types

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
RATE = 16000
CHUNK = int(RATE / 10)
GEMINI_SAMPLE_RATE = 24000

# 1. OPTIMIZATION: Reduced silence limit significantly
# The code was waiting 4 seconds after you stopped talking. Now it waits 1.5s.
SILENCE_LIMIT = 1.5

st.set_page_config(page_title="Umbriel Voice AI", page_icon="🌑")

# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    json_path = st.text_input("Google STT JSON Path", value="speech_totext.json")
    gemini_api_key = st.text_input("Google AI Studio Key", type="password")
    ollama_model = st.selectbox("Ollama Model", ["qwen2.5:1.5b", "llama3.2"], index=0)
    voice_choice = st.selectbox("Voice", ["Umbriel", "Aoede", "Puck", "Charon", "Fenrir", "Despina"])

    st.markdown("---")
    # Increased default sensitivity slightly
    SILENCE_THRESHOLD = st.slider("Silence Threshold", 100, 3000, 500)


# --- Helper: Convert Raw PCM to WAV ---
def pcm_to_wav_bytes(pcm_data):
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(GEMINI_SAMPLE_RATE)
        wav_file.writeframes(pcm_data)
    wav_io.seek(0)
    return wav_io.read()


# --- Audio Playback ---
def play_audio_locally(audio_bytes):
    try:
        wav_data = pcm_to_wav_bytes(audio_bytes)

        # 2. OPTIMIZATION: Check init to avoid overhead
        if pygame.mixer.get_init() is None:
            pygame.mixer.init(frequency=GEMINI_SAMPLE_RATE)

        audio_stream = io.BytesIO(wav_data)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        st.error(f"Audio Error: {e}")


# --- STT Client ---
@st.cache_resource
def init_stt_client(api_key_path):
    try:
        creds = service_account.Credentials.from_service_account_file(api_key_path)
        return speech.SpeechClient(credentials=creds)
    except Exception as e:
        st.error(f"Error loading STT JSON: {e}")
        return None


# --- TTS Client (Gemini) ---
def generate_umbriel_audio(text, api_key, voice_name="Umbriel"):
    if not api_key:
        st.error("Missing API Key")
        return None

    try:
        client = genai.Client(api_key=api_key)

        # 3. OPTIMIZATION: Prompt Engineering for Speed
        # We instruct Gemini to speak naturally but quickly in the content prompt
        # Note: 'speaking_rate' parameter is not fully supported in the generative audio API yet,
        # so we use prompt engineering.
        prompt_text = f"Read the following text clearly but at a slightly fast pace: {text}"

        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                )
            )
        )
        return response.parts[0].inline_data.data

    except Exception as e:
        st.error(f"Umbriel Error: {e}")
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

            # Silence Logic
            if rms < SILENCE_THRESHOLD:
                if start_silence_time is None:
                    start_silence_time = time.time()

                # Uses the new, shorter SILENCE_LIMIT
                if time.time() - start_silence_time > SILENCE_LIMIT:
                    self.closed = True
                    break
            else:
                start_silence_time = None

            yield audio_bytes


# --- Main App ---
st.title("🎙️ Voice AI: Speed Edition")

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

        # 4. OPTIMIZATION: Concise System Prompt
        # We force Ollama to be extremely short.
        # Less text to generate = Faster response time + Faster TTS generation
        llm = ChatOllama(model=ollama_model, temperature=0.7)
        chain = (
                ChatPromptTemplate.from_messages([
                    ("system", "You are a concise voice assistant. Answer in 1 or 2 short sentences max."),
                    ("user", "{input}")
                ]) | llm | StrOutputParser()
        )

        ai_text = chain.invoke({"input": full_transcript})

        st.session_state.messages.append({"role": "assistant", "content": ai_text})
        st.chat_message("assistant").write(ai_text)

        # Speak
        status.update(label=f"Generating Audio...", state="running")
        audio_data = generate_umbriel_audio(ai_text, gemini_api_key, voice_choice)

        if audio_data:
            status.update(label="Speaking", state="complete")
            play_audio_locally(audio_data)