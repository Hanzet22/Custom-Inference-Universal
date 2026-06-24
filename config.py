# config.py
import os
from pathlib import Path

MODEL_DIR = Path("models")
INPUT_DIR = Path("inputs")
RESULT_DIR = Path("results")

# Telegram (ambil dari environment variable, atau set default kosong)
TELEGRAM_BOT_TOKEN = os.getenv("8665685237:AAFj60ckefci50U1st8EJBPYu5FcMaZ28kY", "")
TELEGRAM_CHAT_ID = os.getenv("7307931676", "")

# Default model (untuk fallback jika tidak terdeteksi)
DEFAULT_SCALE = 4
DEFAULT_TILE = 0
DEFAULT_FACE_ENHANCE = False
