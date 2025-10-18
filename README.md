# voice-cloning


🎙️ Voice Cloning 

Clone a person’s voice using **deep learning** — this script captures **Person X’s speech content** (what they said) and re-synthesizes it in **Person Y’s voice** (how they said it).
It uses **OpenAI Whisper** for transcription and **XTTS v2** for multilingual voice cloning.


## 🚀 Features

* 🎧 **Clone voices** with just a few seconds of Person Y’s audio.
* 🧠 **Automatic Speech Recognition (ASR)** using Whisper.
* 🌍 Supports **multiple languages** (English, Malayalam, Hindi, etc.).
* ⚡ GPU-accelerated (Colab or local GPU recommended).
* 🔊 Outputs a **natural, human-like cloned voice**.

---

## 🧩 System Components

| Component              | Model            | Function                                        |
| ---------------------- | ---------------- | ----------------------------------------------- |
| **Speech Recognition** | `openai-whisper` | Converts Person X’s audio to text               |
| **Voice Cloning**      | `XTTS v2`        | Synthesizes Person X’s text in Person Y’s voice |

---

## 🛠️ Setup Instructions

### 1. Install Dependencies

Run these commands first:

```bash
!apt-get install -y espeak-ng ffmpeg
!pip install TTS==0.22.0 openai-whisper
```

---

### 2. Import Libraries

```python
import torch
from TTS.api import TTS
import whisper
from google.colab import files
import IPython.display as ipd
```

---

### 3. Upload Input Files

You’ll need **two audio files**:

* 🎙️ `input_audio.wav` → Person X (the speaker whose *words* you want)
* 🧑‍🏫 `ref_audio.wav` → Person Y (the voice you want to clone)

```python
print("Upload Person X's audio (speech content):")
uploaded_input = files.upload()
input_audio = list(uploaded_input.keys())[0]

print("Upload Person Y's reference voice sample:")
uploaded_ref = files.upload()
ref_audio = list(uploaded_ref.keys())[0]
```

---

### 4. Load XTTS v2 Voice Cloning Model

```python
tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True,
    gpu=torch.cuda.is_available()
)
```

---

### 5. Extract Text from Person X (ASR)

```python
asr_model = whisper.load_model("medium")  # use "large" for even better accuracy
result = asr_model.transcribe(input_audio, language="en")  # or "ml" for Malayalam
person_x_text = result["text"]
print("📝 Extracted text:", person_x_text)
```

---

### 6. Generate Cloned Voice

```python
output_path = "cloned_output.wav"
tts.tts_to_file(
    text=person_x_text,
    speaker_wav=ref_audio,
    language="en",   # change to "ml" for Malayalam
    file_path=output_path
)
```

---

### 7. Play the Output

```python
ipd.Audio(output_path)
```

---

## 🎛️ Example Workflow

| Step             | Input                                                               | Output                |
| ---------------- | ------------------------------------------------------------------- | --------------------- |
| 1️⃣ Upload audio | Person X’s sentence + Person Y’s voice sample                       |                       |
| 2️⃣ Transcribe   | Extract text → “Hello everyone, welcome to my talk.”                |                       |
| 3️⃣ Synthesize   | Cloned voice: Person Y saying “Hello everyone, welcome to my talk.” | ✅ `cloned_output.wav` |

---

## ⚙️ Optional Settings

* Change `language="ml"` for Malayalam or `"hi"` for Hindi.
* Use `whisper.load_model("large")` for higher transcription accuracy.
* Run on **GPU** for faster processing and better TTS results.

---

## 🧾 Output

* File: `cloned_output.wav`
* Content: Person Y’s voice speaking Person X’s words.
* Quality: Near-human, clear pronunciation with emotional tone preserved.

---

## ⚠️ Notes

* Works best with **clean, noise-free recordings**.
* Each audio should be at least **5–10 seconds** long.
* Avoid background music or overlapping voices.
* Colab GPU is **recommended** for real-time inference.

---

## 📚 References

* 🗣️ [Coqui TTS (XTTS v2)](https://github.com/coqui-ai/TTS)
* 🔊 [OpenAI Whisper](https://github.com/openai/whisper)


