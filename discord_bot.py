import discord
import os
import io
import asyncio
import ollama
from google.cloud import speech
from google.cloud import texttospeech
from google.oauth2 import service_account

# --- Configuration ---
TOKEN = "YOUR_DISCORD_BOT_TOKEN_HERE"  # Get this from Discord Developer Portal
GOOGLE_JSON_PATH = "speech_totext.json"
OLLAMA_MODEL = "llama3.2"

# --- Google Clients Setup ---
try:
    creds = service_account.Credentials.from_service_account_file(GOOGLE_JSON_PATH)
    stt_client = speech.SpeechClient(credentials=creds)
    tts_client = texttospeech.TextToSpeechClient(credentials=creds)
    print("Google Clients Loaded Successfully.")
except Exception as e:
    print(f"Error loading Google Credentials: {e}")
    exit(1)

# --- Bot Setup (Use py-cord, not discord.py) ---
intents = discord.Intents.default()
bot = discord.Bot(intents=intents)
connections = {}  # Tracks voice clients per server


# --- Helper Functions ---

async def transcribe_audio(audio_bytes):
    """Sends audio bytes to Google STT and returns text."""
    try:
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,  # Discord audio is 48k
            language_code="en-US",
            audio_channel_count=2  # Discord is stereo
        )
        response = stt_client.recognize(config=config, audio=audio)

        # Combine all results
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        return transcript
    except Exception as e:
        print(f"STT Error: {e}")
        return ""


async def get_ai_response(text):
    """Sends text to Ollama and returns AI response."""
    if not text: return None
    try:
        response = ollama.generate(model=OLLAMA_MODEL, prompt=text + " reply in 40 words")
        return response['response']
    except Exception as e:
        return f"Ollama Error: {e}"


async def generate_audio(text):
    """Converts AI text to Audio Bytes using Google TTS."""
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Journey-F",  # High quality voice
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    # Note: Discord expects 48k stereo, but we can stream standard MP3/Opus
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return response.audio_content


# --- Bot Commands ---

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.slash_command(name="join", description="Join your voice channel")
async def join(ctx):
    voice = ctx.author.voice
    if not voice:
        return await ctx.respond("You aren't in a voice channel!")

    vc = await voice.channel.connect()
    connections.update({ctx.guild.id: vc})
    await ctx.respond("Joined! Use `/listen` to talk to me.")


@bot.slash_command(name="listen", description="Record for 5 seconds and reply")
async def listen(ctx):
    if ctx.guild.id not in connections:
        return await ctx.respond("I am not in a voice channel. Use `/join` first.")

    vc = connections[ctx.guild.id]

    # Callback function required by py-cord
    async def finished_callback(sink, channel: discord.TextChannel, *args):
        # 1. Get audio for the user who spoke
        recorded_users = [
            f"<@{user_id}>" for user_id, audio in sink.audio_data.items()
        ]

        # We only care about the user who ran the command (or the first audio stream)
        # sink.audio_data is a dict: {user_id: AudioData}
        if not sink.audio_data:
            await channel.send("I didn't hear anything.")
            return

        # Grab the first available audio stream
        user_id = list(sink.audio_data.keys())[0]
        audio_file = sink.audio_data[user_id].file

        # Audio comes as raw PCM/WAV bytes in the file object
        audio_bytes = audio_file.read()

        await channel.send(f"Processing audio from <@{user_id}>...")

        # 2. Transcribe
        user_text = await transcribe_audio(audio_bytes)
        if not user_text:
            await channel.send("Could not understand audio.")
            return

        await channel.send(f"🗣️ **You said:** {user_text}")

        # 3. Think (Ollama)
        ai_reply = await get_ai_response(user_text)
        await channel.send(f"🤖 **AI:** {ai_reply}")

        # 4. Speak (TTS)
        audio_response_bytes = await generate_audio(ai_reply)

        # Play back to Discord
        # We create a BytesIO source for discord to play
        source = discord.FFmpegPCMAudio(
            io.BytesIO(audio_response_bytes),
            pipe=True,
            options="-vn"  # No video
        )
        vc.play(source)

    # Start recording audio from the channel
    # WaveSink records to standard WAV format which Google STT likes
    vc.start_recording(
        discord.sinks.WaveSink(),
        finished_callback,
        ctx.channel
    )

    await ctx.respond("Listening for 5 seconds... 🔴")
    await asyncio.sleep(5)  # Record for 5 seconds
    vc.stop_recording()  # This triggers the 'finished_callback'


@bot.slash_command(name="leave", description="Leave the voice channel")
async def leave(ctx):
    if ctx.guild.id in connections:
        vc = connections[ctx.guild.id]
        await vc.disconnect()
        del connections[ctx.guild.id]
        await ctx.respond("Goodbye!")
    else:
        await ctx.respond("I'm not connected.")


bot.run(TOKEN)