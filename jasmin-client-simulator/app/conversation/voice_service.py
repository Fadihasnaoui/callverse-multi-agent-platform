import io
import base64
import logging
from typing import Optional
from gtts import gTTS
from app.core.config import settings
from app.core.exceptions import LLMProviderException

logger = logging.getLogger("simulator")

class VoiceService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "recording.webm", language: str = "fr") -> str:
        """
        Transcribe audio input using Groq Whisper API (whisper-large-v3-turbo).
        """
        if not self.api_key or settings.LLM_PROVIDER.lower() != "groq":
            logger.warning("Groq API key missing or provider not set to Groq. Cannot perform live Whisper transcription.")
            return ("Hello, I'd like some information about the offers available." if language == "en"
                    else "Bonjour, je souhaite vous demander des informations sur les offres disponibles.")

        try:
            from groq import Groq
            client = Groq(api_key=self.api_key)
            
            # Create in-memory file object with appropriate filename
            audio_file = (filename, io.BytesIO(audio_bytes))
            
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                language=language,  # the advisor speaks the simulation language: better accuracy than auto-detect
                response_format="json"
            )
            
            transcript_text = transcription.text.strip()
            logger.info(f"Groq Whisper transcription success: '{transcript_text}'")
            return transcript_text  # empty string = silence, caller decides
        except Exception as e:
            logger.error(f"Whisper Transcription failed: {str(e)}")
            raise LLMProviderException(f"Échec de la transcription audio : {str(e)}")


    def synthesize_speech_base64(self, text: str, lang: str = "fr") -> str:
        """
        Synthesize text response to MP3 audio formatted as base64 data URI.
        """
        try:
            tts = gTTS(text=text, lang=lang, slow=False)  # lang = simulation language (fr | en)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            audio_b64 = base64.b64encode(fp.read()).decode("utf-8")
            return f"data:audio/mp3;base64,{audio_b64}"
        except Exception as e:
            logger.error(f"Text-to-Speech Error: {str(e)}")
            return ""

voice_service = VoiceService()
