import whisper

def extract_text_from_audio(audio_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error processing Audio {audio_path}: {e}")
        return ""
