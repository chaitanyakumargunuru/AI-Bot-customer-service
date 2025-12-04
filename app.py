import streamlit as st
import queue
import sys
import io
import re
import time  # <--- Added for timing silence
import audioop  # <--- Added for calculating audio volume
import pygame
import ollama
import pyaudio
from google.cloud import speech
from google.cloud import texttospeech
from google.oauth2 import service_account

llm_model = "qwen2.5:1.5b"
# --- Page Config ---
st.set_page_config(page_title="Voice AI Assistant", page_icon="🎙️")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Configuration Constants ---
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks

# --- Silence Detection Settings ---
SILENCE_THRESHOLD = 700  # Amplitude level. Below this is considered "silence".
# If it stops too early, INCREASE this. If it never stops, DECREASE this.
SILENCE_LIMIT = 4  # Seconds of silence before stopping

# --- Sidebar Settings ---
with st.sidebar:
    st.title("Settings")
    json_path = st.text_input("Google JSON Path", value="speech_totext.json")
    ollama_model = st.selectbox("Ollama Model", ["llama3.2", "qwen2.5:1.5b", "mistral"], index=0)

    # Optional: Let user adjust sensitivity in sidebar
    st.markdown("---")
    st.markdown("**Mic Sensitivity**")
    SILENCE_THRESHOLD = st.slider("Silence Threshold", 100, 3000, 700, help="Increase if background is noisy")
    st.info("Note: Ensure 'ollama serve' is running.")


# --- Helper Functions ---

def clean_text_for_tts(text):
    cleaned = re.sub(r'[\*#_]', '', text)
    cleaned = " ".join(cleaned.split())
    return cleaned


def play_audio_locally(audio_bytes):
    try:
        pygame.mixer.init()
        audio_stream = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        st.error(f"Audio Error: {e}")


@st.cache_resource
def init_clients(api_key_path):
    try:
        creds = service_account.Credentials.from_service_account_file(api_key_path)
        stt_client = speech.SpeechClient(credentials=creds)
        tts_client = texttospeech.TextToSpeechClient(credentials=creds)
        ollama_client = ollama.Client()
        return stt_client, tts_client, ollama_client
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return None, None, None


def text_to_audio_bytes(text_input, tts_client):
    synthesis_input = texttospeech.SynthesisInput(text=text_input)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-IN",
        name="en-IN-Journey-O",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=0.90
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    return response.audio_content


# --- Microphone Stream Class (Modified for Silence Detection) ---
class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
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
        # Silence detection variables
        start_silence_time = None
        has_spoken_yet = False  # Optional: prevent stopping before user starts speaking

        while not self.closed:
            # 1. Get Data
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

            # Combine chunks for processing
            audio_bytes = b"".join(data)

            # 2. Calculate Volume (RMS)
            # 2 bytes width because paInt16
            rms = audioop.rms(audio_bytes, 2)

            # 3. Silence Logic
            if rms < SILENCE_THRESHOLD:
                if start_silence_time is None:
                    start_silence_time = time.time()

                # Check if silence limit reached
                duration = time.time() - start_silence_time
                if duration > SILENCE_LIMIT:
                    # Only stop if we have actually started speaking or if we want strict timeout
                    # Here we stop strictly after 4s silence regardless
                    print("\nSilence limit reached. Stopping...")
                    self.closed = True
                    break
            else:
                # Noise detected, reset silence timer
                start_silence_time = None
                has_spoken_yet = True

            yield audio_bytes


# --- Main Logic ---

st.title("🎙️ AI Voice Assistant")
st.caption(f"Auto-stops after {SILENCE_LIMIT} seconds of silence.")

# Load Clients
stt_client, tts_client, ollama_client = init_clients(json_path)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Recording Control
col1, col2 = st.columns([1, 4])
with col1:
    start_btn = st.button("Start Recording")

if start_btn and stt_client:
    transcript_placeholder = st.empty()
    status_text = st.status(f"Listening... (Will stop after {SILENCE_LIMIT}s silence)", expanded=True)

    transcript_parts = []
    full_transcript = ""

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    try:
        # The stream will now close itself when silence is detected
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_generator = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)

            # This loop finishes when stream.generator() breaks (due to silence)
            responses = stt_client.streaming_recognize(config=streaming_config, requests=requests)

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                current_transcript = result.alternatives[0].transcript.strip()

                # Update UI
                transcript_placeholder.info(f"🗣️ {current_transcript}")

                if result.is_final:
                    transcript_parts.append(current_transcript)

        # Reconstruct full text
        full_transcript = " ".join(transcript_parts).strip()

        if full_transcript:
            status_text.update(label="Processing Answer...", state="running")

            # 1. User Message
            st.session_state.messages.append({"role": "user", "content": full_transcript})
            with st.chat_message("user"):
                st.markdown(full_transcript)

            transcript_placeholder.empty()

            # 2. Generate AI Response
            prompt_text = full_transcript + " give answer in 50 words"

            try:
                llm_response = ollama_client.generate(model=ollama_model, prompt=prompt_text)
                ai_text = llm_response['response']

                # 3. AI Message Display
                st.session_state.messages.append({"role": "assistant", "content": ai_text})
                with st.chat_message("assistant"):
                    st.markdown(ai_text)

                    # 4. Generate Audio
                    clean_ai_text = clean_text_for_tts(ai_text)
                    audio_bytes = text_to_audio_bytes(clean_ai_text, tts_client)

                    # --- NEW: Save audio file in background ---
                    # This creates a unique file (e.g., response_171000.mp3)
                    filename = f"response_{int(time.time())}.mp3"

                    # "wb" means Write Binary (needed for audio files)
                    with open(filename, "wb") as f:
                        f.write(audio_bytes)

                    print(f"Saved audio locally to: {filename}")
                    # ------------------------------------------

                    # 5. Play Audio (Autoplay)
                    play_audio_locally(audio_bytes)

            except Exception as e:
                st.error(f"AI/TTS Error: {e}")

        else:
            status_text.warning("No speech detected.")

    except Exception as e:
        # Pyaudio might throw an error when we force close, we can ignore specific ones
        if "Stream closed" not in str(e):
            st.error(f"Stream Error: {e}")