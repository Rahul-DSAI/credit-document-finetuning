import json
import uuid
import time
import asyncio
import logging
from statistics import mean
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
openai_api_key = os.getenv("OPENAI_API_KEY")  # replace or use env var
client = OpenAI(api_key=openai_api_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Evolution Dashboard Backend")

# ─── In‐Memory Stores ───────────────────────────────────────────────────────────
DATASETS: Dict[str, List[Dict[str, Any]]] = {}
TEST_RESULTS: Dict[str, Dict[str, Any]] = {}

# ─── Schemas ───────────────────────────────────────────────────────────────────
class DatasetInfo(BaseModel):
    dataset_id: str
    num_records: int

class ModelTestRequest(BaseModel):
    model_id: str
    dataset_id: str

class ModelTestResult(BaseModel):
    test_id: str
    model_id: str
    dataset_id: str
    num_tests: int
    successes: int
    failures: int
    accuracy: float
    metrics: Dict[str, Dict[str, float]]
    details: List[Dict[str, Any]]

# ─── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/datasets/upload", response_model=DatasetInfo)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a JSONL dataset. Each line must be:
      {"messages": [...], "expected": "..."}
    """
    dataset_id = str(uuid.uuid4())
    raw = await file.read()
    lines = raw.decode("utf-8").splitlines()

    records = []
    for idx, line in enumerate(lines):
        try:
            data = json.loads(line)
            if "messages" not in data or "expected" not in data:
                raise ValueError("Missing 'messages' or 'expected'")
            records.append(data)
        except Exception as e:
            raise HTTPException(400, f"Line {idx+1} invalid: {e}")

    DATASETS[dataset_id] = records
    logger.info(f"Dataset {dataset_id} uploaded with {len(records)} records")
    return DatasetInfo(dataset_id=dataset_id, num_records=len(records))


@app.get("/models/list")
async def list_models():
    """
    Return list of available OpenAI chat-capable models.
    """
    resp = client.models.list()
    # resp.data is a list of Model objects with an .id attribute
    models = [m.id for m in resp.data if "gpt-" in m.id or "turbo" in m.id]
    return {"models": models}


async def run_single_test(model_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoke a chat completion, compare to expected, and capture usage & latency.
    """
    start = time.monotonic()
    try:
        resp = await client.chat.completions.acreate(
            model=model_id,
            messages=record["messages"],
            temperature=0.0,
            max_tokens=200
        )
        latency = time.monotonic() - start

        output = resp.choices[0].message.content.strip()
        success = (output == record["expected"].strip())

        usage = resp.usage
        return {
            "prompt": record["messages"],
            "expected": record["expected"],
            "output": output,
            "success": success,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "latency_s": latency
        }
    except Exception as e:
        latency = time.monotonic() - start
        return {
            "prompt": record["messages"],
            "expected": record["expected"],
            "output": None,
            "success": False,
            "error": str(e),
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "latency_s": latency
        }


@app.post("/models/test", response_model=ModelTestResult)
async def test_model(req: ModelTestRequest):
    """
    Run all records in a dataset through the given model.
    Returns a summary with successes, accuracy, and aggregated metrics.
    """
    if req.dataset_id not in DATASETS:
        raise HTTPException(404, "Dataset not found")

    records = DATASETS[req.dataset_id]
    test_id = str(uuid.uuid4())

    tasks = [run_single_test(req.model_id, rec) for rec in records]
    results = await asyncio.gather(*tasks)

    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    failures = total - successes
    accuracy = successes / total if total else 0.0

    # Collect numeric metrics
    latencies       = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    prompt_toks     = [r["prompt_tokens"] for r in results if r.get("prompt_tokens") is not None]
    completion_toks = [r["completion_tokens"] for r in results if r.get("completion_tokens") is not None]
    total_toks      = [r["total_tokens"] for r in results if r.get("total_tokens") is not None]

    summary_metrics = {
        "latency_s": {
            "avg": mean(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "prompt_tokens": {
            "avg": mean(prompt_toks) if prompt_toks else None,
            "total": sum(prompt_toks) if prompt_toks else None,
        },
        "completion_tokens": {
            "avg": mean(completion_toks) if completion_toks else None,
            "total": sum(completion_toks) if completion_toks else None,
        },
        "total_tokens": {
            "avg": mean(total_toks) if total_toks else None,
            "total": sum(total_toks) if total_toks else None,
        },
    }

    summary = {
        "test_id": test_id,
        "model_id": req.model_id,
        "dataset_id": req.dataset_id,
        "num_tests": total,
        "successes": successes,
        "failures": failures,
        "accuracy": accuracy,
        "metrics": summary_metrics,
        "details": results
    }

    TEST_RESULTS[test_id] = summary
    logger.info(
        f"Test {test_id} for model {req.model_id}: "
        f"{successes}/{total} successes, accuracy={accuracy:.2%}"
    )
    return summary


@app.get("/tests/{test_id}", response_model=ModelTestResult)
async def get_test_result(test_id: str):
    """
    Retrieve a previous test’s results.
    """
    if test_id not in TEST_RESULTS:
        raise HTTPException(404, "Test ID not found")
    return TEST_RESULTS[test_id]


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,           # Import string for auto-reload
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )