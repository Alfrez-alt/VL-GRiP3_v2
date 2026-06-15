import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import whisper
import os
import logging

from .paths import WHISPER_DIR

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, output_dir=None):
        self.output_dir = str(output_dir) if output_dir is not None else str(WHISPER_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        self.wav_path = os.path.join(self.output_dir, "prompt.wav")
        # You can later add additional configuration parameters as needed.

    def record_and_transcribe(self, duration=5, samplerate=16000, channels=1):
        # --- Step 1: List devices
        print("\n🎧 Available audio devices:\n")
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"[{idx}] {dev['name']}")

        # --- Step 2: Select input device
        try:
            device_id = int(input("\n🔧 Enter the device ID for your microphone: "))
            sd.default.device = device_id
        except Exception as e:
            logger.error("Error selecting device: %s", e)
            return ""

        # --- Step 3: Record from the selected device
        logger.info("Recording... Speak now!")
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate,
                       channels=channels, dtype='int16')
        sd.wait()
        logger.info("Recording done.")

        # --- Step 4: Save WAV file
        write(self.wav_path, samplerate, audio)
        logger.info("Audio saved at: %s", self.wav_path)

        # --- Step 5: Transcribe using Whisper
        logger.info("Loading Whisper model...")
        model = whisper.load_model("turbo")  # Options: "tiny", "small", "medium", "large", "turbo"
        result = model.transcribe(self.wav_path, language="en", temperature=0.0, task="transcribe")
        transcription = result["text"].strip()
        logger.info("Transcription: %s", transcription)
        return transcription
