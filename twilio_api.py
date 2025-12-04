import streamlit as st
import threading
import queue
import json
import base64
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
import pyaudio
import audioop
import time
from google.cloud import speech, texttospeech
from google.oauth2 import service_account
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
ST_PORT = 8501
FASTAPI_PORT = 8000
CHUNK = 1600
RATE = 16000  # Local Mic Rate
PHONE_RATE = 8000  # Twilio Rate (Standard)

st.set_page_config(page_title="Unified AI Agent", layout="wide")

# --- GLOBAL QUEUES (The bridge between Phone & UI) ---
phone_log_queue = queue.Queue()
phone_status_queue = queue.Queue()


# --- HELPER FUNCTIONS (Google & LangChain) ---
@st.cache_resource
def init_google():
    # Ensure speech_totext.json is in your folder
    creds = service_account.Credentials.from_service_account_file("speech_totext.json")
    stt = speech.SpeechClient(credentials=creds)
    tts = texttospeech.TextToSpeechClient(credentials=creds)
    return stt, tts


def get_ai_chain():
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.7)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Concise voice assistant (under 30 words)."),
        ("user", "{input}")
    ])
    return prompt | llm | StrOutputParser()


# --- FASTAPI SERVER (Background Phone Logic) ---
app = FastAPI()


@app.post("/voice")
async def voice(request: Request):
    """Twilio Webhook: Instructions when call connects"""
    host = request.headers.get("host")
    # TwiML to connect audio stream
    xml = f"""
    <Response>
        <Say>Connecting you to the A.I. Dashboard.</Say>
        <Connect>
            <Stream url="wss://{host}/media-stream" />
        </Connect>
    </Response>
    """
    return HTMLResponse(content=xml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    """Handle Audio Stream from Phone"""
    await websocket.accept()
    phone_status_queue.put("📞 Phone Call Connected!")

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
        sample_rate_hertz=8000,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    async def request_generator():
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                if data['event'] == 'media':
                    chunk = base64.b64decode(data['media']['payload'])
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                elif data['event'] == 'stop':
                    break
            except:
                break

    try:
        # Note: In production, use SpeechAsyncClient for better async performance
        requests = request_generator()
        responses = client.streaming_recognize(config=streaming_config, requests=requests)

        for response in responses:
            if not response.results: continue
            result = response.results[0]
            if not result.alternatives: continue

            transcript = result.alternatives[0].transcript
            # Send transcript to Streamlit UI
            if result.is_final:
                phone_log_queue.put(f"📱 Caller: {transcript}")
            else:
                pass  # Ignore interim results for logs to keep it clean

    except Exception as e:
        phone_status_queue.put(f"Connection Error: {e}")

    phone_status_queue.put("🔴 Call Ended")


def run_fastapi_thread():
    """Function to run uvicorn in a thread"""
    uvicorn.run(app, host="0.0.0.0", port=FASTAPI_PORT, log_level="error")


# --- START BACKGROUND THREAD ---
if "server_running" not in st.session_state:
    t = threading.Thread(target=run_fastapi_thread, daemon=True)
    t.start()
    st.session_state.server_running = True

# --- STREAMLIT UI ---
st.title("🤖 Unified AI Command Center")

stt_client, tts_client = init_google()
ai_chain = get_ai_chain()

# Create Tabs
tab1, tab2 = st.tabs(["🎙️ Local Mic Mode", "📞 Phone Server Logs"])

# === TAB 1: LOCAL MODE (Your Original Code) ===
with tab1:
    st.subheader("Talk to AI via Computer Mic")
    if st.button("Start Local Recording"):
        st.info("Listening... (Simulated for brevity in this unified view)")
        # [Insert your MicrophoneStream Logic here]
        # For this example, I'll simulate a result so the code runs without your specific Mic class
        user_input = "Hello AI"
        response = ai_chain.invoke({"input": user_input})
        st.write(f"**You:** {user_input}")
        st.success(f"**AI:** {response}")

# === TAB 2: PHONE MODE (The Dashboard) ===
with tab2:
    st.subheader("Live Phone Call Feed")

    # Instructions
    with st.expander("Setup Info"):
        st.code(f"ngrok http {FASTAPI_PORT}", language="bash")
        st.write("Point Twilio Webhook to: `https://<ngrok-url>/voice`")

    # Status Indicator
    status_box = st.empty()

    # Log Container
    log_container = st.container()

    # Initialize session state for logs
    if "phone_logs" not in st.session_state:
        st.session_state.phone_logs = []

    # Check Queues (Poll for data from background thread)
    try:
        while not phone_status_queue.empty():
            status_msg = phone_status_queue.get_nowait()
            st.toast(status_msg)

        while not phone_log_queue.empty():
            log_msg = phone_log_queue.get_nowait()
            st.session_state.phone_logs.append(log_msg)

    except queue.Empty:
        pass

    # Display Logs
    with log_container:
        for log in st.session_state.phone_logs:
            st.markdown(f"> {log}")

    # Auto-refresh mechanism for Tab 2
    if st.checkbox("Auto-refresh logs", value=True):
        time.sleep(1)
        st.rerun()