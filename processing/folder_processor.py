import os
from langchain.schema import Document as LCDocument
from config.settings import INPUT_FOLDER
from extractors.pdf_extractor import extract_text_from_pdf
from extractors.image_extractor import extract_text_from_image
from extractors.docx_extractor import extract_text_from_docx
from extractors.audio_extractor import extract_text_from_audio
from extractors.text_extractor import extract_text_from_txt

def process_folder(input_folder=INPUT_FOLDER):
    extracted_texts = []
    
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        file_extension = filename.lower().split('.')[-1]
        
        extracted_text = ""
        if file_extension == "pdf":
            extracted_text = extract_text_from_pdf(file_path)
        elif file_extension in ["png", "jpg", "jpeg", "tiff", "bmp", "gif"]:
            extracted_text = extract_text_from_image(file_path)
        elif file_extension == "docx":
            extracted_text = extract_text_from_docx(file_path)
        elif file_extension == "txt":
            extracted_text = extract_text_from_txt(file_path)
        elif file_extension in ["mp3", "wav", "m4a", "flac"]:
            extracted_text = extract_text_from_audio(file_path)
        
        if extracted_text:
            extracted_texts.append(LCDocument(page_content=extracted_text, metadata={"filename": filename}))
            print(f"Processed: {filename}")
        else:
            print(f"Skipping empty or unsupported file: {filename}")

    return extracted_texts
