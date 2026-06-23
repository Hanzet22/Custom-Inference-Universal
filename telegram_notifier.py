# telegram_notifier.py
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

def send_error_to_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        # Jika tidak dikonfigurasi, lewati
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": f"❌ Error Inference:\n{message}"}
        requests.post(url, data=data, timeout=5)
    except Exception:
        pass  # Abaikan jika gagal kirim
