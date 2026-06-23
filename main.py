# main.py
import argparse
import sys
from pathlib import Path
from config import MODEL_DIR, INPUT_DIR, RESULT_DIR
from detector import ensure_dirs, scan_models, inspect_file, print_report
from loader import load_model_universal
from upscaler import upscale_image, upscale_video
from telegram_notifier import send_error_to_telegram

def main():
    parser = argparse.ArgumentParser(description="Universal Model Inspector & Upscaler")
    parser.add_argument("--mode", choices=["inspect", "upscale"], default="inspect",
                        help="Pilih mode: inspect (deteksi) atau upscale (jalankan upscale)")
    parser.add_argument("--models", default=str(MODEL_DIR), help="Folder model")
    parser.add_argument("--model", default="", help="File model spesifik")
    # Argumen untuk inspect
    parser.add_argument("--unsafe-torchload", action="store_true", help="Izinkan torch.load untuk inspeksi")
    # Argumen untuk upscale
    parser.add_argument("--input", default=str(INPUT_DIR), help="Input gambar atau video (file/folder)")
    parser.add_argument("--output", default=str(RESULT_DIR), help="Folder output")
    parser.add_argument("--scale", type=int, default=4, help="Skala upscale")
    parser.add_argument("--tile", type=int, default=0, help="Tile size (0 = off)")
    parser.add_argument("--face_enhance", action="store_true", help="Gunakan GFPGAN untuk face enhance")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID")
    args = parser.parse_args()

    ensure_dirs()
    model_dir = Path(args.models)

    # Tentukan model files
    if args.model:
        model_files = [Path(args.model)] if Path(args.model).exists() else []
        if not model_files:
            print(f"Model {args.model} tidak ditemukan.")
            sys.exit(1)
    else:
        model_files = scan_models(model_dir)
        if not model_files:
            print(f"Tidak ada model di {model_dir}")
            sys.exit(0)

    if args.mode == "inspect":
        for p in model_files:
            try:
                report = inspect_file(p, unsafe_torchload=args.unsafe_torchload)
                print_report(report)
            except Exception as e:
                print(f"Error inspecting {p}: {e}")
                send_error_to_telegram(f"Inspect error {p}: {e}")
    else:  # upscale
        # Untuk setiap model, lakukan upscale pada semua input
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input {input_path} tidak ditemukan.")
            sys.exit(1)
        # Load model pertama (atau bisa dipilih)
        model_path = model_files[0] if model_files else None
        if not model_path:
            print("Tidak ada model untuk dijalankan.")
            sys.exit(1)
        print(f"Memuat model: {model_path}")
        try:
            model, scale = load_model_universal(model_path)
            # Override scale jika args.scale diberikan
            if args.scale:
                scale = args.scale
        except Exception as e:
            print(f"Gagal memuat model: {e}")
            send_error_to_telegram(f"Load model error {model_path}: {e}")
            sys.exit(1)

        # Kumpulkan file input
        if input_path.is_file():
            input_files = [input_path]
        else:
            input_files = sorted(input_path.glob("*"))

        for file in input_files:
            if not file.is_file():
                continue
            ext = file.suffix.lower()
            output_name = f"{file.stem}_out{file.suffix}"
            output_path = Path(args.output) / output_name
            try:
                if ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    print(f"Upscale video: {file}")
                    upscale_video(model, file, output_path, scale=args.scale, tile=args.tile,
                                  face_enhance=args.face_enhance, gpu_id=args.gpu_id)
                else:
                    print(f"Upscale image: {file}")
                    upscale_image(model, file, output_path, scale=args.scale, tile=args.tile,
                                  face_enhance=args.face_enhance, gpu_id=args.gpu_id)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                send_error_to_telegram(f"Upscale error {file}: {e}")

if __name__ == "__main__":
    main()
