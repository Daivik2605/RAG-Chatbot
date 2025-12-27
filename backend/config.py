import os
from dotenv import load_dotenv
from pathlib import Path

# Explicitly load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = "vectorstore"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environment")
