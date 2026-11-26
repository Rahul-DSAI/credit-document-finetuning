import json
import re
from config.settings import STRUCTURED_QA_JSON, JSONL_FILE

def clean_question(question):
    """
    Cleans the question by removing leading numbers, #, *, and excessive whitespace.
    """
    # Remove leading numbers (e.g., "1. ", "2) ")
    question = re.sub(r'^[\d]+[).\-*\s]*', '', question)
    # Remove leading/trailing special characters
    question = re.sub(r'^[#*\s]+|[#*\s]+$', '', question)
    return question.strip()

def save_json(data, filename=STRUCTURED_QA_JSON):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Structured data saved to {filename}")

def convert_to_jsonl(input_json=STRUCTURED_QA_JSON, output_jsonl=JSONL_FILE):
    with open(input_json, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for qa in qa_data:
            cleaned_question = clean_question(qa["question"])
            formatted_entry = {
                "messages": [
                    {"role": "system", "content": "The Credit Assistant chatbot handles credit-related queries"},
                    {"role": "user", "content": cleaned_question},
                    {"role": "assistant", "content": qa["answer"]}
                ]
            }
            f.write(json.dumps(formatted_entry, ensure_ascii=False) + "\n")
    
    print(f"Formatted Q&A saved to {output_jsonl} in JSONL format!")