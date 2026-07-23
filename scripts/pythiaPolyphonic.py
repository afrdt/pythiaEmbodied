import os
import signal
import numpy as np
import sounddevice as sd
import scipy.io.wavfile
import subprocess
import requests
import soundfile as sf
import re
from datetime import datetime

# ── RAG imports ───────────────────────────────────────────────────────────────
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_PATH   = "/home/pythia/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL  = "/home/pythia/whisper.cpp/models/ggml-tiny.en.bin"
AUDIO_FILE     = "/home/pythia/input.wav"
OLLAMA_MODEL   = "oracle"
SAMPLE_RATE    = 16000

# Three voices with different timbres — subtle pitch shifts only
VOICE_AMY      = "/home/pythia/piper_voices/en_US-amy-medium.onnx"
VOICE_JENNY    = "/home/pythia/piper_voices/en_GB-jenny-medium.onnx"
VOICE_LESSAC   = "/home/pythia/piper_voices/en_US-lessac-medium.onnx"

# Subtle pitch shifts — just enough to customize, not cartoonify
PITCH_AMY      = 0.944   # 1 semitone down
PITCH_JENNY    = 1.0     # no shift — keep her natural
PITCH_LESSAC   = 0.971   # half semitone down

REPLIES_DIR    = "/home/pythia/rag_project/replies"

# RAG settings
DATASET_PATH   = "/home/pythia/rag_project/pythiaDatasetClean.txt"
CHROMA_DIR     = "/home/pythia/chroma_db"
CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 100
TOP_K          = 2

# Recording
MAX_RECORD_SECONDS = 15
SILENCE_THRESHOLD  = 400
SILENCE_SECONDS    = 3
CHUNK_DURATION     = 0.3

# Ollama
MAX_TOKENS = 120

# ── RAG: chunking & indexing ──────────────────────────────────────────────────
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        start += size - overlap
    return chunks

def build_vector_store(dataset_path, chroma_dir):
    print("Initialising vector store...")
    embedder   = ONNXMiniLM_L6_V2()
    client     = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        "pythia_docs",
        embedding_function=embedder
    )

    existing = collection.count()
    if existing > 0:
        print(f"Vector store ready ({existing} chunks already indexed).")
        return collection, embedder

    print("Reading dataset...")
    with open(dataset_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Embedding {len(chunks)} chunks — this will take a few minutes on first run...")

    batch_size = 256
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        ids   = [f"chunk_{start + i}" for i in range(len(batch))]
        collection.add(documents=batch, ids=ids)
        print(f"  Indexed {min(start + batch_size, len(chunks))}/{len(chunks)}")

    print("Indexing complete.")
    return collection, embedder

def retrieve_context(query, collection, embedder, top_k=TOP_K):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]

# ── Recording with silence detection ─────────────────────────────────────────
def record_audio(filename):
    print("Listening...")
    frames = []
    silent_chunks  = 0
    max_chunks     = int(MAX_RECORD_SECONDS / CHUNK_DURATION)
    required_quiet = int(SILENCE_SECONDS / CHUNK_DURATION)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(int(CHUNK_DURATION * SAMPLE_RATE))
            frames.append(chunk.copy())
            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
            silent_chunks = silent_chunks + 1 if rms < SILENCE_THRESHOLD else 0
            if silent_chunks >= required_quiet and len(frames) > required_quiet + 2:
                break

    audio = np.concatenate(frames, axis=0)
    scipy.io.wavfile.write(filename, SAMPLE_RATE, audio)
    print(f"Captured {len(frames) * CHUNK_DURATION:.1f}s of audio.")

# ── Whisper transcription ─────────────────────────────────────────────────────
def transcribe_whisper(audio_file):
    result = subprocess.run(
        [WHISPER_PATH, "-m", WHISPER_MODEL, "-f", audio_file, "-otxt", "-of", "transcript"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Whisper error:", result.stderr)
        return None

    if not os.path.exists("transcript.txt"):
        return None

    with open("transcript.txt", "r") as f:
        text = f.read().strip()
    os.remove("transcript.txt")

    if not text:
        return None

    print("You said:", text)
    return text

# ── Ollama query with RAG context ─────────────────────────────────────────────
def query_ollama(prompt, context_chunks):
    print("Thinking...")
    context_block = "\n\n".join(context_chunks)

    full_prompt = (
        f"Draw upon these sacred texts to inform your prophecy. "
        f"If they are not relevant, speak from your own ancient wisdom. "
        f"Do not quote them directly — let their wisdom flow through your words.\n\n"
        f"[SACRED TEXTS]\n{context_block}\n[/SACRED TEXTS]\n\n"
        f"Seeker's question: {prompt}"
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": MAX_TOKENS,
                    "temperature": 0.7,
                }
            },
            timeout=60
        )
    except requests.exceptions.Timeout:
        return "The vision is unclear. Ask again."
    except requests.exceptions.ConnectionError:
        return "The oracle cannot speak right now."

    if response.status_code != 200:
        print("Ollama error:", response.text)
        return "The oracle is silent."

    output = response.json()["response"].strip()
    print("Oracle:", output)
    return output

# ── Piper TTS — three voice chorus ───────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def generate_voice(text, model, output_file):
    result = subprocess.run(
        ["piper", "--model", model, "--output_file", output_file],
        input=text,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Piper error ({model}): {result.stderr}")
        return False
    return os.path.exists(output_file)

def pitch_shift(input_file, output_file, pitch):
    """Apply subtle pitch shift via asetrate."""
    if pitch == 1.0:
        subprocess.run(["cp", input_file, output_file])
        return os.path.exists(output_file)
    subprocess.run([
        "ffmpeg", "-y", "-i", input_file,
        "-af", f"asetrate=22050*{pitch},aresample=22050",
        output_file
    ], capture_output=True)
    return os.path.exists(output_file)

def mix_voices(amy, jenny, lessac, output):
    """Mix three voices with glue reverb and volume boost."""
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", amy, "-i", jenny, "-i", lessac,
        "-filter_complex",
        "amix=inputs=3:duration=longest:normalize=0:weights=1 0.8 0.7,"
        "aecho=0.8:0.7:40:0.2,"
        "volume=3.0",
        output
    ], capture_output=True, text=True)
    return os.path.exists(output)

def speak(text):
    cleaned = clean_text(text)
    if len(cleaned) < 5:
        print("Response too short to speak.")
        return

    print("Speaking...")
    tmp = {
        "amy_raw":    "/tmp/amy_raw.wav",
        "jenny_raw":  "/tmp/jenny_raw.wav",
        "lessac_raw": "/tmp/lessac_raw.wav",
        "amy":        "/tmp/amy.wav",
        "jenny":      "/tmp/jenny.wav",
        "lessac":     "/tmp/lessac.wav",
        "mixed":      "/tmp/mixed.wav",
    }

    try:
        # Generate all three voices
        ok1 = generate_voice(cleaned, VOICE_AMY,    tmp["amy_raw"])
        ok2 = generate_voice(cleaned, VOICE_JENNY,  tmp["jenny_raw"])
        ok3 = generate_voice(cleaned, VOICE_LESSAC, tmp["lessac_raw"])

        if not (ok1 and ok2 and ok3):
            print("One or more voices failed.")
            return

        # Apply subtle pitch shifts
        pitch_shift(tmp["amy_raw"],    tmp["amy"],    PITCH_AMY)
        pitch_shift(tmp["jenny_raw"],  tmp["jenny"],  PITCH_JENNY)
        pitch_shift(tmp["lessac_raw"], tmp["lessac"], PITCH_LESSAC)

        # Mix with reverb glue and volume boost
        if not mix_voices(tmp["amy"], tmp["jenny"], tmp["lessac"], tmp["mixed"]):
            print("Voice mixing failed.")
            return

        # Save to replies folder
        os.makedirs(REPLIES_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(REPLIES_DIR, f"reply_poly_{timestamp}.wav")
        subprocess.run(["cp", tmp["mixed"], save_path])
        print(f"Saved: {save_path}")

        # Play
        data, samplerate = sf.read(tmp["mixed"])
        sd.play(data, samplerate)
        sd.wait()

    except Exception as e:
        print(f"TTS error: {e}")
    finally:
        for f in tmp.values():
            if os.path.exists(f):
                os.remove(f)

# ── Utilities ─────────────────────────────────────────────────────────────────
def normalize_prompt(prompt):
    return re.sub(r"[^\w\s]", "", prompt).strip().lower()

def cleanup():
    for f in [AUDIO_FILE, "transcript.txt", "response.wav"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    def _handle_sigint(sig, frame):
        print("\nFarewell.")
        cleanup()
        raise SystemExit(0)
    signal.signal(signal.SIGINT, _handle_sigint)

    os.makedirs(CHROMA_DIR, exist_ok=True)
    collection, embedder = build_vector_store(DATASET_PATH, CHROMA_DIR)

    print("\nPythia is listening. Say 'farewell' to exit.\n")
    while True:
        record_audio(AUDIO_FILE)
        prompt = transcribe_whisper(AUDIO_FILE)
        if not prompt:
            continue

        if normalize_prompt(prompt) == "farewell":
            speak("Farewell, seeker.")
            cleanup()
            break

        chunks = retrieve_context(prompt, collection, embedder)
        print("RAG:", [c[:80] for c in chunks])
        response = query_ollama(prompt, chunks)
        speak(response)

if __name__ == "__main__":
    main()
