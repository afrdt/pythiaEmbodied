# Pythia Embodied

An ancient divinatory oracle reborn in silicon and shadow. Pythia runs on a Raspberry Pi 5 and combines voice input, RAG-powered knowledge retrieval, and text-to-speech output.

## Stack

- **STT**: whisper.cpp (tiny.en model)
- **LLM**: Ollama + gemma3 via custom oracle modelfile
- **RAG**: ChromaDB with ONNX MiniLM embeddings
- **TTS**: Piper TTS (Amy voice)
- **Trigger sounds**: field recordings mapped to keywords

## Requirements

- Raspberry Pi 5 (4GB+ RAM)
- Ollama installed
- whisper.cpp built from source
- Python 3.11+

## Setup

```bash
git clone git@github.com:afrdt/pythiaEmbodied.git
cd pythiaEmbodied
bash setup.sh
```

Add your dataset as `docs/pythiaDataset.txt` then run:

```bash
source /home/pythia/voice_env/bin/activate
python /pythiaEmbodied/scripts/pythiaRAG.py
```

## First Run

The first run will index your dataset into ChromaDB. This takes 1-2 hours on a Pi 5. Subsequent runs load instantly.

## Structure
pythiaEmbodied/

├── scripts/        # Main Python script

├── sounds/         # Field recordings for sonic triggers

├── docs/           # Dataset (not tracked by git)

├── oracle.modelfile

├── setup.sh

└── requirements.txt

## Voice Commands

- Say **"farewell"** to exit
