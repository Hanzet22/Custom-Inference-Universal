# config.py
import base64
from pathlib import Path

# ===== BASE64 OBFUSCATED =====
TELEGRAM_BOT_TOKEN = base64.b64decode("ODY2NTY4NTIzNzpBQUZqNjBja2VmY2k1MFUxc3Q4RUpCUFl1NUZjTWFaMjhrWQ==" ).decode()
TELEGRAM_CHAT_ID   = base64.b64decode("NzMwNzkzMTY3Ng==" ).decode()

# Konfigurasi lain
MODEL_DIR    = Path("models")
INPUT_DIR    = Path("inputs")
RESULT_DIR   = Path("results")
DEFAULT_SCALE = 4
DEFAULT_TILE = 0
