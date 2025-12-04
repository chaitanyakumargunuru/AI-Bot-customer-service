import queue
import sys
import io
import re
import time
import pygame
import ollama
import pyaudio
from google.cloud import speech
from google.cloud import texttospeech
from google.oauth2 import service_account


def clean_text_for_tts(text):
    cleaned = re.sub(r'[\*#_]', '', text)

    # Optional: Remove extra whitespace created by replacements
    cleaned = " ".join(cleaned.split())
    return cleaned

# --- Configuration ---
# 1. Credentials
CLIENT_FILE = "speech_totext.json"
try:
    CREDENTIALS = service_account.Credentials.from_service_account_file(CLIENT_FILE)
except FileNotFoundError:
    print(f"Error: Could not find '{CLIENT_FILE}'. Please verify the file path.")
    sys.exit(1)

# 2. Audio Recording Parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms chunks

# 3. Ollama Model
OLLAMA_MODEL = "llama3.2"

# 4. Command to stop
STOP_COMMAND = "stop recording"


# --- TTS Function (Modified for Speed) ---
def speak_text_directly(text_input, tts_client):
    """
    Synthesizes speech using a pre-initialized client.
    This eliminates the handshake/auth delay on every request.
    """
    try:
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

        # Use the passed client (no re-initialization)
        response = tts_client.synthesize_speech(
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


# --- Microphone Stream Class (Infinite Stream) ---
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
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def request_generator(audio_stream):
    for content in audio_stream:
        yield speech.StreamingRecognizeRequest(audio_content=content)


# --- Main Logic ---
def main():
    # 1. Initialize Clients ONCE (Global Optimization)
    print("Initializing Google Clients...", end=" ")
    stt_client = speech.SpeechClient(credentials=CREDENTIALS)

    # This is the key fix: Create the TTS client here, not inside the loop
    tts_client_global = texttospeech.TextToSpeechClient(credentials=CREDENTIALS)

    ollama_client = ollama.Client()
    print("Done.")

    # STT Config
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False
    )

    print(f"\n--- AI Voice Assistant ({OLLAMA_MODEL}) Initialized ---")
    print(f"Mode: Records until you say '{STOP_COMMAND}'")

    while True:
        try:
            print(f"\nListening... (Say '{STOP_COMMAND}' to finish)", flush=True)

            transcript_parts = []

            with MicrophoneStream(RATE, CHUNK) as stream:
                audio_generator = stream.generator()
                requests = request_generator(audio_generator)

                responses = stt_client.streaming_recognize(config=streaming_config, requests=requests)

                for response in responses:
                    if not response.results:
                        continue

                    result = response.results[0]
                    if not result.alternatives:
                        continue

                    current_transcript = result.alternatives[0].transcript.strip()

                    sys.stdout.write(f"\rCaptured: {current_transcript}")
                    sys.stdout.flush()

                    if STOP_COMMAND in current_transcript.lower():
                        print(f"\n\nCommand '{STOP_COMMAND}' detected. Stopping...")
                        clean_part = current_transcript.lower().replace(STOP_COMMAND, "")
                        if result.is_final:
                            transcript_parts.append(clean_part)
                        break

                    if result.is_final:
                        transcript_parts.append(current_transcript)
                        sys.stdout.write("\n")

            user_text = " ".join(transcript_parts).strip() +" give answer in 50 words"

            if user_text:
                print(f"\nFinal Input sent to AI: {user_text}")
                print("Thinking...", flush=True)

                try:
                    response = ollama_client.generate(model=OLLAMA_MODEL, prompt=user_text)
                    llm_output = response['response']

                    print(f"AI: {llm_output}")

                    # --- NEW: Clean the text before speaking ---
                    clean_audio_text = clean_text_for_tts(llm_output)

                    # Pass the CLEANED text to the function
                    speak_text_directly(clean_audio_text, tts_client_global)

                except Exception as e:
                    print(f"Ollama Error: {e}")
            else:
                print("\nNo speech detected before stop command.")

        except KeyboardInterrupt:
            print("\nStopping Program...")
            break
        except Exception as e:
            print(f"Critical Error: {e}")
            break


if __name__ == "__main__":
    main()