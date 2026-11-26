import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Paths
INPUT_FOLDER = "data/input"
OUTPUT_FOLDER = "data/output"
JSONL_FILE = os.path.join(OUTPUT_FOLDER, "formatted_qa.jsonl")
STRUCTURED_QA_JSON = os.path.join(OUTPUT_FOLDER, "structured_qa.json")
