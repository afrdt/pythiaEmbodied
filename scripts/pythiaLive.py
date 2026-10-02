import os
import signal
import numpy as np
import sounddevice as sd
import scipy.io.wavfile
import subprocess
import requests
import soundfile as sf
import re

# ── RAG imports ───────────────────────────────────────────────────────────────
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_PATH   = "/home/pythia/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL  = "/home/pythia/whisper.cpp/models/ggml-tiny.en.bin"
AUDIO_FILE     = "/home/pythia/input.wav"
OLLAMA_MODEL   = "oracle"
SAMPLE_RATE    = 16000

PIPER_MODEL    = "/home/pythia/piper_voices/en_US-amy-medium.onnx"
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
        f"Remember: you are Pythia. Respond as Pythia only.\nSeeker's question: {prompt}"
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

# ── Piper TTS ─────────────────────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r"\*+", "", text)          # strip asterisks
    text = re.sub(r"#+\s*", "", text)         # strip markdown headers
    text = re.sub(r"_{2,}", "", text)         # strip underscores
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)  # strip links
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def speak(text):
    cleaned = clean_text(text)
    if len(cleaned) < 5:
        print("Response too short to speak.")
        return

    print("Speaking...")
    try:
        result = subprocess.run(
            ["piper", "--model", PIPER_MODEL, "--output_file", "response.wav"],
            input=cleaned,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("Piper error:", result.stderr)
            return

        if os.path.exists("response.wav"):
            # Boost volume
            subprocess.run([
                "ffmpeg", "-y", "-i", "response.wav",
                "-af", "asetrate=22050*0.75,aresample=22050,volume=15.0",
                "response_boosted.wav"
            ], capture_output=True)
            if os.path.exists("response_boosted.wav"):
                os.replace("response_boosted.wav", "response.wav")

            sd.play(data, samplerate)
            sd.wait()
            os.remove("response.wav")
        else:
            print("Piper did not create response.wav.")
    except FileNotFoundError:
        print("Piper not found — is it installed and on PATH?")
    except Exception as e:
        print(f"TTS error: {e}")

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
