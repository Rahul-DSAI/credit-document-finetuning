from processing.folder_processor import process_folder
from processing.qa_extractor import extract_questions_answers
from processing.json_formatter import save_json, convert_to_jsonl
from models.fine_tuning import upload_jsonl_file, fine_tune_model
from config.settings import JSONL_FILE

def main():
    print("Starting pipeline execution...")

    # Step 1: Process documents
    extracted_docs = process_folder()
    print("Documents processed.")

    # Step 2: Extract Q&A
    structured_qas = extract_questions_answers(extracted_docs)
    print("Questions and answers extracted.")

    # Step 3: Save structured JSON
    save_json(structured_qas)
    print(f"Structured data saved to {JSONL_FILE}")

    # Step 4: Convert to JSONL
    convert_to_jsonl()
    print(f"JSONL file generated.")

    # Step 5: Upload training data
    jsonl_file_id = upload_jsonl_file(JSONL_FILE)
    print(f"File uploaded successfully! File ID: {jsonl_file_id}")

    # Step 6: Fine-tune model
    print("Starting fine-tuning process...")
    model_id = fine_tune_model(jsonl_file_id)

    if model_id:
        print(f"Fine-tuning completed! Model ID: {model_id}")
    else:
        print("Fine-tuning failed. Please check OpenAI logs.")

if __name__ == "__main__":
    main()
