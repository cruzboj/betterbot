# Project - BetterBot
docker compose up -d

for virtual mechine users or none graphics cards user use...
http://127.0.0.1:8888/tree

Remove-Item -Recurse -Force .venv

uv venv
uv init
requires-python = "==3.12.*"

uv add lm_polygraph transformers accelerate ipykernel

[X] uv add --index https://download.pytorch.org/whl/cu124 torch torchvision torchaudio 
[X] uv add --index https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
uv add --index https://download.pytorch.org/whl/cu130 torch torchvision torchaudio

uv add spacy
uv run python -m spacy download en_core_web_sm

Get-ChildItem -Path .\.venv -Recurse -File | Unblock-File

source .venv/bin/activate


# install wordnet LCA / Lowest Common Ancestor
uv add nltk

# delete models
huggingface-cli delete-cache


