def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            return txt_file.read()
    except Exception as e:
        print(f"Error processing TXT {txt_path}: {e}")
        return ""
