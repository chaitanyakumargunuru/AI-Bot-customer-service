import streamlit as st
import queue
import re
import time
import audioop
import io
import pygame
import pyaudio
from google.cloud import speech
from google.cloud import texttospeech
from google.oauth2 import service_account

# --- LangChain Imports ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="Voice AI (LangChain)", page_icon="🦜")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Configuration ---
RATE = 16000
CHUNK = int(RATE / 10)
SILENCE_THRESHOLD = 700
SILENCE_LIMIT = 4

# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    json_path = st.text_input("Google JSON Path", value="speech_totext.json")
    ollama_model = st.selectbox("Ollama Model", ["qwen2.5:1.5b", "llama3.2", "mistral"], index=0)

    st.markdown("---")
    SILENCE_THRESHOLD = st.slider("Silence Threshold", 100, 3000, 700)
    st.info("Ensure 'ollama serve' is running.")


# --- Helper Functions ---
def clean_text_for_tts(text):
    cleaned = re.sub(r'[\*#_]', '', text)
    return " ".join(cleaned.split())


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
def init_google_clients(api_key_path):
    try:
        creds = service_account.Credentials.from_service_account_file(api_key_path)
        stt = speech.SpeechClient(credentials=creds)
        tts = texttospeech.TextToSpeechClient(credentials=creds)
        return stt, tts
    except Exception as e:
        st.error(f"Error loading Google creds: {e}")
        return None, None


def get_langchain_chain(model_name):
    """Initializes the LangChain LLM Chain."""
    # 1. Define the LLM
    llm = ChatOllama(model=model_name, temperature=0.7)

    # 2. Define the Prompt
    # This replaces manually adding "give answer in 50 words" strings
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful voice assistant. Keep your answers concise and short (under 50 words)."),
        ("user", "{input}")
    ])

    # 3. Create the Chain (Prompt -> LLM -> Text Output)
    chain = prompt | llm | StrOutputParser()
    return chain


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


# --- Microphone Stream Class (Unchanged) ---
class MicrophoneStream:
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


# --- Main Logic ---

st.title("🦜 LangChain Voice Assistant")

# Initialize Clients
stt_client, tts_client = init_google_clients(json_path)
# Initialize LangChain
ai_chain = get_langchain_chain(ollama_model)

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

col1, col2 = st.columns([1, 4])
with col1:
    start_btn = st.button("Start Recording")

if start_btn and stt_client:
    transcript_placeholder = st.empty()
    status = st.status("Listening...", expanded=True)

    full_transcript = ""
    transcript_parts = []

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    try:
        with MicrophoneStream(RATE, CHUNK) as stream:
            audio_gen = stream.generator()
            requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_gen)
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

        if full_transcript:
            status.update(label="Thinking (LangChain)...", state="running")
            transcript_placeholder.empty()

            # 1. User Message
            st.session_state.messages.append({"role": "user", "content": full_transcript})
            with st.chat_message("user"):
                st.markdown(full_transcript)

            # 2. AI Generation via LangChain
            try:
                # --- LangChain Execution ---
                # We invoke the chain with the user input
                ai_text_stream = ai_chain.stream({"input": full_transcript})

                # Stream the text response to UI
                full_ai_response = ""
                with st.chat_message("assistant"):
                    response_container = st.empty()
                    for chunk in ai_text_stream:
                        full_ai_response += chunk
                        response_container.markdown(full_ai_response + "▌")
                    response_container.markdown(full_ai_response)

                st.session_state.messages.append({"role": "assistant", "content": full_ai_response})

                # 3. TTS Generation
                status.update(label="Generating Audio...", state="running")
                clean_text = clean_text_for_tts(full_ai_response)
                audio_bytes = text_to_audio_bytes(clean_text, tts_client)

                # Save & Play
                # filename = f"response_{int(time.time())}.mp3"
                # with open(filename, "wb") as f:
                #     f.write(audio_bytes)

                status.update(label="Finished", state="complete")
                play_audio_locally(audio_bytes)

            except Exception as e:
                st.error(f"LangChain Error: {e}")

    except Exception as e:
        if "Stream closed" not in str(e):
            st.error(f"Stream Error: {e}")