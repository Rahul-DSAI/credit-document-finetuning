from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error processing Image {image_path}: {e}")
        return ""
