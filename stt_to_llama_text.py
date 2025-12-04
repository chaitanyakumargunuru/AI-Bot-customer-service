import queue
import sys
import io
import time
import math
import struct  # Replaces audioop for unpacking audio bytes
import pygame
import ollama
import pyaudio
from google.cloud import speech
from google.cloud import texttospeech
from google.oauth2 import service_account

# --- Configuration ---
# 1. Credentials
CLIENT_FILE = "speech_totext.json"  # Ensure this file is in your folder
try:
    CREDENTIALS = service_account.Credentials.from_service_account_file(CLIENT_FILE)
except FileNotFoundError:
    print(f"Error: Could not find '{CLIENT_FILE}'. Please verify the file path.")
    sys.exit(1)

# 2. Audio Recording Parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks

# 3. Silence Detection Settings
SILENCE_LIMIT = 15  # Seconds of silence before stopping
SILENCE_THRESHOLD = 500  # Volume threshold (Increase if background noise triggers it)

# 4. Ollama Model
OLLAMA_MODEL = "llama3.2"


# --- Helper: Calculate RMS (Volume) without audioop ---
def calculate_rms(audio_data):
    """Calculates RMS amplitude of 16-bit audio data manually."""
    if not audio_data:
        return 0

    # Calculate how many 16-bit integers we have (2 bytes per integer)
    count = len(audio_data) // 2

    # Unpack the bytes into a tuple of integers ('h' is for short/16-bit signed)
    shorts = struct.unpack(f"{count}h", audio_data)

    # Calculate sum of squares
    sum_squares = sum(s ** 2 for s in shorts)

    # Calculate RMS: sqrt(mean of squares)
    rms = math.sqrt(sum_squares / count)
    return int(rms)


# --- TTS Function ---
def speak_text_directly(text_input, credentials):
    """Synthesizes speech and plays it directly from memory using Pygame."""
    try:
        client = texttospeech.TextToSpeechClient(credentials=credentials)

        synthesis_input = texttospeech.SynthesisInput(text=text_input)

        # 'Journey' is a conversational voice.
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Journey-F",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Play directly from RAM
        pygame.mixer.init()
        audio_stream = io.BytesIO(response.audio_content)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()

        # Wait for audio to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        print(f"TTS Error: {e}")


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
        last_sound_time = time.time()

        while not self.closed:
            # Blocking get to wait for data
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Consume any other buffered data
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            # Combine chunks
            audio_data = b"".join(data)

            # --- SILENCE DETECTION LOGIC ---
            # Replaced audioop.rms with manual calculation
            rms = calculate_rms(audio_data)

            if rms > SILENCE_THRESHOLD:
                last_sound_time = time.time()  # Reset timer because we heard sound
            else:
                # Check how long it has been silent
                if time.time() - last_sound_time > SILENCE_LIMIT:
                    print(f"\n[Silence detected for {SILENCE_LIMIT} seconds. Processing...]")
                    return  # This stops the generator and closes the Google stream

            yield audio_data


# --- Helper Function for Request Generation ---
def request_generator(audio_stream):
    # NOTE: We do NOT yield the config here because we pass it directly
    # to stt_client.streaming_recognize(config=..., requests=...)
    # The client library automatically sends the config as the first request.
    for content in audio_stream:
        yield speech.StreamingRecognizeRequest(audio_content=content)


# --- Main Logic ---
def main():
    # Initialize Clients
    stt_client = speech.SpeechClient(credentials=CREDENTIALS)
    ollama_client = ollama.Client()

    # STT Config
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )

    # IMPORTANT: single_utterance=False allows long recording sessions
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False,
        single_utterance=False
    )

    print(f"\n--- AI Voice Assistant ({OLLAMA_MODEL}) Initialized ---")
    print(f"Mode: Records until {SILENCE_LIMIT} seconds of silence is detected.")

    while True:
        try:
            print("\nListening... (Speak now)", flush=True)

            with MicrophoneStream(RATE, CHUNK) as stream:
                audio_generator = stream.generator()
                requests = request_generator(audio_generator)

                # Call Google API (this blocks until silence limit is reached)
                responses = stt_client.streaming_recognize(config=streaming_config, requests=requests)

                # Combine all partial transcripts into one full sentence
                full_transcript = []
                for response in responses:
                    if not response.results:
                        continue
                    for result in response.results:
                        if result.is_final:
                            transcript = result.alternatives[0].transcript
                            full_transcript.append(transcript)
                            print(f"Captured part: {transcript}")

                user_text = " ".join(full_transcript).strip()

            # Process with LLM & Speak
            if user_text:
                print(f"\nFinal User Input: {user_text}")
                print("Thinking...", flush=True)

                try:
                    response = ollama_client.generate(model=OLLAMA_MODEL, prompt=user_text)
                    llm_output = response['response']

                    print(f"AI: {llm_output}")
                    speak_text_directly(llm_output, CREDENTIALS)
                except Exception as e:
                    print(f"Ollama Error: {e}")
            else:
                print("\nNo speech detected.")

        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Critical Error: {e}")
            break


if __name__ == "__main__":
    main()