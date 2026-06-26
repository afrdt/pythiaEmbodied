#!/bin/bash
echo "Setting up Pythia..."

# Create venv
python3 -m venv /home/pythia/voice_env
source /home/pythia/voice_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model
ollama create oracle -f oracle.modelfile

# Download piper voice
mkdir -p /home/pythia/piper_voices
wget -nc https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx -O /home/pythia/piper_voices/en_US-amy-medium.onnx
wget -nc https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json -O /home/pythia/piper_voices/en_US-amy-medium.onnx.json

echo "Done. Add your dataset to docs/ then run: python scripts/pythiaRAG.py"
