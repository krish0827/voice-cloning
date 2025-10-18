

# ----------------------------
# Voice Cloning with XTTS v2
# ----------------------------

# Install dependencies
!apt-get install -y espeak-ng ffmpeg
!pip install TTS==0.22.0 openai-whisper

# Imports
import torch
from TTS.api import TTS
import whisper
from google.colab import files
import IPython.display as ipd

# ----------------------------
# Upload files
# ----------------------------
print("Upload Person X's audio (speech content):")
uploaded_input = files.upload()
input_audio = list(uploaded_input.keys())[0]

print("Upload Person Y's reference voice sample:")
uploaded_ref = files.upload()
ref_audio = list(uploaded_ref.keys())[0]

# ----------------------------
# Load XTTS v2 (more accurate than YourTTS)
# ----------------------------
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True,
    gpu=torch.cuda.is_available()
)

# ----------------------------
# Extract text from Person X (ASR)
# ----------------------------
asr_model = whisper.load_model("medium")  # try "large" for even better accuracy
result = asr_model.transcribe(input_audio, language="en")  # or "ml" for Malayalam
person_x_text = result["text"]
print("📝 Extracted text:", person_x_text)

# ----------------------------
# Generate cloned voice (Person X's words in Person Y's voice)
# ----------------------------
output_path = "cloned_output.wav"
tts.tts_to_file(
    text=person_x_text,
    speaker_wav=ref_audio,
    language="en",   # change to "ml" for Malayalam
    file_path=output_path
)

# Play audio
ipd.Audio(output_path)
